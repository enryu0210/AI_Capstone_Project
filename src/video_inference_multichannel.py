"""
video_inference_multichannel.py
================================
4채널 YOLOv8n-cls 모델을 이용한 실제 영상 추론 파이프라인.

기존 3채널 추론(video_inference.ipynb)과의 차이:
  1. 4번째 채널: 이전 프레임과의 grayscale diff
     → 학습 데이터(smoke-clean diff)와 도메인이 비슷해 얇은 연기 탐지에 유리
  2. hybrid_smoke_decision: YOLO confidence + 영상 품질 지표 결합
     → video_inference.ipynb 셀 0에 스케치된 로직을 완전 구현
     → 얇은 연기(YOLO conf 낮음)도 선명도/대비/색상 균일화 지표로 보완 탐지
  3. Sliding window vote는 기존과 동일하게 유지

출력:
  runs/smoke_detector/multichannel_video/
    capture1_result.avi   ← 오버레이 영상
    smoke_timeline.png    ← 5초 단위 smoke 비율 차트
    frame_results.csv     ← 프레임별 판정 결과
"""

import sys
import time
import csv
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경 대응
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from ultralytics import YOLO


# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "assets" / "capture1.avi"
MODEL_PATH = BASE_DIR / "runs" / "smoke_detector" / "multichannel" / "best.pt"
OUT_DIR    = BASE_DIR / "runs" / "smoke_detector" / "multichannel_video"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO  = OUT_DIR / "capture1_result.avi"

# ── 추론 파라미터 ──────────────────────────────────────────
IMGSZ         = 224
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"

# Sliding Window Vote (기존 3채널과 동일)
VOTE_WINDOW   = 5
VOTE_K        = 3

# Hybrid Decision 파라미터
# video_inference.ipynb 셀 0의 hybrid_smoke_decision 스케치에서 발전
WARMUP_FRAMES = 30    # 기준값(baseline) 산출에 사용할 초기 프레임 수
HIGH_CONF_THR = 0.60  # YOLO 확신 임계값: 이 이상이면 즉시 smoke 판정
MID_CONF_THR  = 0.25  # 중간 임계값: haze_score와 결합 시 사용
HAZE_DEGRADE  = 0.70  # 품질 저하 기준: baseline 대비 70% 이하이면 "저하" 판정


# ────────────────────────────────────────────────────────────
# 1. 모델 구성 (train_multichannel.py와 동일한 아키텍처)
# ────────────────────────────────────────────────────────────

def build_4ch_model() -> nn.Module:
    """
    YOLOv8n-cls의 첫 번째 Conv를 4채널로 수정한 모델 아키텍처를 반환한다.
    train_multichannel.py와 동일한 구조 (가중치는 별도로 로드).
    """
    yolo       = YOLO("yolov8n-cls.pt")
    nn_model   = yolo.model
    first_conv = nn_model.model[0].conv

    new_conv = nn.Conv2d(
        in_channels  = 4,
        out_channels = first_conv.out_channels,
        kernel_size  = first_conv.kernel_size,
        stride       = first_conv.stride,
        padding      = first_conv.padding,
        bias         = first_conv.bias is not None,
    )

    # 가중치 초기화 (추후 best.pt로 덮어씌워지므로 형태만 맞추면 됨)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight.data.clone()
        new_conv.weight[:, 3:, :, :] = first_conv.weight.data.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    nn_model.model[0].conv = new_conv
    return nn_model


def load_model() -> nn.Module:
    """학습된 best.pt 가중치를 로드하여 4채널 모델을 반환한다."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 가중치 없음: {MODEL_PATH}\n"
            "먼저 train_multichannel.py 를 실행하세요."
        )
    model = build_4ch_model()
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    print(f"모델 로드 완료: {MODEL_PATH}")
    return model


# ────────────────────────────────────────────────────────────
# 2. 영상 품질 지표 계산
# ────────────────────────────────────────────────────────────

def compute_frame_features(frame_bgr: np.ndarray) -> dict:
    """
    영상 품질 관련 3가지 지표를 계산한다.
    연기 발생 시 나타나는 변화:
      sharpness (선명도) ↓  : 연기로 인해 경계가 흐려짐
      contrast  (대비)   ↓  : 밝고 어두운 구분이 줄어듦
      ch_std    (채색)   ↓  : B/G/R 채널이 비슷해져 뿌옇게 됨
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian variance: 높을수록 선명한 경계 많음
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # grayscale 표준편차: 높을수록 명암 대비 뚜렷
    contrast = float(gray.std())

    # BGR 채널별 평균의 표준편차: 낮을수록 R/G/B가 비슷 → 뿌연 상태
    ch_means = frame_bgr.astype(np.float32).mean(axis=(0, 1))
    ch_std   = float(ch_means.std())

    return {"sharpness": sharpness, "contrast": contrast, "ch_std": ch_std}


# ────────────────────────────────────────────────────────────
# 3. Baseline 캘리브레이션 (Warmup)
# ────────────────────────────────────────────────────────────

def calibrate_baseline(video_path: Path, n_frames: int = WARMUP_FRAMES) -> dict:
    """
    영상 첫 n_frames 프레임의 품질 지표를 측정해 "정상(no-smoke) 기준값"을 산출한다.
    이후 각 프레임의 품질이 이 기준에서 얼마나 벗어났는지 비교한다.

    threshold_c = mean_contrast * HAZE_DEGRADE  (대비 하한선)
    threshold_s = mean_ch_std   * HAZE_DEGRADE  (채색 균일화 하한선)
    baseline_sharpness = mean_sharpness          (sharp_ratio 계산 기준)
    """
    cap = cv2.VideoCapture(str(video_path))
    sharpnesses, contrasts, ch_stds = [], [], []
    count = 0

    while count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        feats = compute_frame_features(frame)
        sharpnesses.append(feats["sharpness"])
        contrasts.append(feats["contrast"])
        ch_stds.append(feats["ch_std"])
        count += 1

    cap.release()

    baseline = {
        "sharpness"  : float(np.mean(sharpnesses)),
        "threshold_c": float(np.mean(contrasts)) * HAZE_DEGRADE,
        "threshold_s": float(np.mean(ch_stds))   * HAZE_DEGRADE,
    }
    print(
        f"[캘리브레이션] baseline_sharpness={baseline['sharpness']:.1f}  "
        f"threshold_c={baseline['threshold_c']:.2f}  "
        f"threshold_s={baseline['threshold_s']:.2f}  "
        f"(첫 {count}프레임 기반)"
    )
    return baseline


# ────────────────────────────────────────────────────────────
# 4. Hybrid Smoke Decision
#    video_inference.ipynb 셀 0 스케치의 완전 구현
# ────────────────────────────────────────────────────────────

def hybrid_smoke_decision(
    yolo_conf: float,
    features: dict,
    baseline: dict,
) -> tuple:
    """
    YOLO confidence + 영상 품질 지표를 결합한 하이브리드 판정.

    판정 로직 (video_inference.ipynb 셀 0 스케치 기반):
      케이스 1: YOLO 확신 높음(≥ HIGH_CONF_THR=0.60)
                → 바로 smoke  (두꺼운 연기, 명확한 케이스)
      케이스 2: YOLO 애매(≥ MID_CONF_THR=0.25) + 품질 저하 2개 이상
                → smoke  (얇은 연기: YOLO는 애매하지만 영상이 뿌옇게 변함)
      케이스 3: YOLO 낮지만 품질 저하 지표 3개 모두 충족
                → smoke_warning  (YOLO는 놓쳤지만 영상 품질 급격 저하)
      나머지   → no_smoke

    Args:
        yolo_conf: smoke 클래스의 softmax 확률 (0~1)
        features : compute_frame_features() 반환값
        baseline : calibrate_baseline() 반환값

    Returns:
        (decision, haze_score)
          decision  : "smoke" | "smoke_warning" | "no_smoke"
          haze_score: 충족된 품질 저하 지표 수 (0~3, 오버레이 표시용)
    """
    sharpness   = features["sharpness"]
    contrast    = features["contrast"]
    ch_std      = features["ch_std"]

    baseline_sharpness = baseline["sharpness"]
    threshold_c        = baseline["threshold_c"]
    threshold_s        = baseline["threshold_s"]

    # 기준 대비 현재 선명도 비율 (낮을수록 흐릿)
    sharp_ratio = sharpness / max(baseline_sharpness, 1e-6)

    # 케이스 1: YOLO 확신 높음 → 즉시 smoke
    if yolo_conf >= HIGH_CONF_THR:
        haze_score = sum([
            sharp_ratio < HAZE_DEGRADE,
            contrast    < threshold_c,
            ch_std      < threshold_s,
        ])
        return "smoke", haze_score

    # 품질 저하 지표 집계
    haze_score = 0
    if sharp_ratio < HAZE_DEGRADE:  # 선명도 30% 이상 하락
        haze_score += 1
    if contrast < threshold_c:      # 대비 기준치 이하
        haze_score += 1
    if ch_std < threshold_s:        # 색상 균일화 (뿌옇게)
        haze_score += 1

    # 케이스 2: YOLO 중간 + 품질 저하 2개 이상 → smoke
    if yolo_conf >= MID_CONF_THR and haze_score >= 2:
        return "smoke", haze_score

    # 케이스 3: YOLO 낮지만 품질 저하 지표 강력 → 경고
    if haze_score >= 3:
        return "smoke_warning", haze_score

    return "no_smoke", haze_score


# ────────────────────────────────────────────────────────────
# 5. 프레임 전처리
# ────────────────────────────────────────────────────────────

def preprocess_frame(
    frame_bgr: np.ndarray,
    prev_gray: np.ndarray,
    imgsz: int = IMGSZ,
) -> tuple:
    """
    BGR 프레임 → 4채널 텐서 (1, 4, imgsz, imgsz).
    ch 0~2: RGB  /  ch 3: 이전 프레임과의 grayscale diff

    학습 시 diff = smoke - clean GT (정적)
    추론 시 diff = 현재 - 이전 프레임 (동적, 움직임 + 연기 변화)
    → 학습 데이터 생성 시 노이즈를 추가해 이 도메인 갭을 줄임

    Returns:
        tensor  : (1, 4, imgsz, imgsz) float32 텐서
        cur_gray: 다음 프레임의 prev_gray로 사용할 현재 grayscale
    """
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, prev_gray) if prev_gray is not None else np.zeros_like(gray)

    # 리사이즈
    rgb_r  = cv2.resize(rgb,  (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    diff_r = cv2.resize(diff, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # 정규화 [0,255] → [0,1]
    rgb_norm  = rgb_r.astype(np.float32)  / 255.0
    diff_norm = diff_r.astype(np.float32) / 255.0

    # (H, W, 4) → (1, 4, H, W)
    img_4ch = np.dstack([rgb_norm, diff_norm[:, :, np.newaxis]])  # (H, W, 4)
    tensor  = torch.FloatTensor(img_4ch.transpose(2, 0, 1)).unsqueeze(0)  # (1, 4, H, W)

    return tensor, gray


# ────────────────────────────────────────────────────────────
# 6. 오버레이 그리기
# ────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    final_label: str,
    yolo_conf: float,
    vote_count: int,
    haze_score: int,
    frame_idx: int,
    total: int,
    elapsed: float,
) -> np.ndarray:
    """상단 바에 최종 판정, YOLO confidence, haze 지표를 표시한다."""
    h, w = frame.shape[:2]

    # 판정별 색상
    if final_label == "smoke_warning":
        bar_color = (0, 165, 255)   # 주황 (경고)
    elif final_label == "smoke":
        bar_color = (0, 0, 200)     # 빨강
    else:
        bar_color = (0, 180, 0)     # 초록

    cv2.rectangle(frame, (0, 0), (w, 65), bar_color, -1)

    # 최종 판정 텍스트
    status_map = {
        "smoke"        : f"[SMOKE]       vote={vote_count}/{VOTE_WINDOW}",
        "smoke_warning": f"[SMOKE WARN]  vote={vote_count}/{VOTE_WINDOW}",
        "no_smoke"     : f"[NO SMOKE]    vote={vote_count}/{VOTE_WINDOW}",
    }
    cv2.putText(frame, status_map.get(final_label, ""), (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # YOLO conf + haze score (보조 정보)
    raw_text = f"YOLO conf={yolo_conf:.2f}  haze={haze_score}/3"
    cv2.putText(frame, raw_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # 진행 정보 (우측 상단)
    info = f"Frame {frame_idx}/{total}  |  {elapsed:.1f}s"
    cv2.putText(frame, info, (w - 280, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ────────────────────────────────────────────────────────────
# 7. 메인 추론 루프
# ────────────────────────────────────────────────────────────

def main():
    print(f"디바이스 : {DEVICE}")
    print(f"입력 영상: {VIDEO_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"영상 파일 없음: {VIDEO_PATH}")

    # 영상 기본 정보
    cap          = cv2.VideoCapture(str(VIDEO_PATH))
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS          = cap.get(cv2.CAP_PROP_FPS)
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"해상도: {W}×{H_}  FPS: {FPS:.2f}  총 {TOTAL_FRAMES}프레임 ({TOTAL_FRAMES/FPS:.1f}초)")

    # 기준값 캘리브레이션 (첫 30프레임으로 정상 상태 기준 산출)
    baseline = calibrate_baseline(VIDEO_PATH)

    # 모델 로드
    model = load_model()

    # 추론 루프 초기화
    cap      = cv2.VideoCapture(str(VIDEO_PATH))
    writer   = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"XVID"), FPS, (W, H_))
    vote_buf = deque(maxlen=VOTE_WINDOW)

    smoke_frames  = 0
    ms_list       = []
    frame_idx     = 0
    log_interval  = 500
    start_wall    = time.perf_counter()
    prev_gray     = None
    frame_results = []  # (frame_idx, final_label, yolo_conf) — 타임라인 생성용

    print(f"\n추론 시작 ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ① 영상 품질 지표 계산
        features = compute_frame_features(frame)

        # ② 4채널 전처리 + 모델 추론
        t0 = time.perf_counter()
        tensor, cur_gray = preprocess_frame(frame, prev_gray)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            logits     = model(tensor)
            probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
            yolo_conf  = float(probs[1])  # smoke 클래스 index=1

        ms_list.append((time.perf_counter() - t0) * 1000)
        prev_gray = cur_gray

        # ③ Hybrid 판정 (YOLO conf + 품질 지표)
        raw_label, haze_score = hybrid_smoke_decision(yolo_conf, features, baseline)

        # ④ Sliding Window Vote (단발성 오탐 제거)
        vote_buf.append(1 if raw_label in ("smoke", "smoke_warning") else 0)
        vote_count  = sum(vote_buf)
        final_label = raw_label if vote_count >= VOTE_K else "no_smoke"

        if final_label in ("smoke", "smoke_warning"):
            smoke_frames += 1

        frame_results.append((frame_idx, final_label, yolo_conf))

        # ⑤ 오버레이 및 저장
        elapsed   = time.perf_counter() - start_wall
        out_frame = draw_overlay(
            frame.copy(), final_label, yolo_conf,
            vote_count, haze_score, frame_idx, TOTAL_FRAMES, elapsed
        )
        writer.write(out_frame)

        if frame_idx % log_interval == 0 or frame_idx == TOTAL_FRAMES:
            pct    = frame_idx / TOTAL_FRAMES * 100
            avg_ms = np.mean(ms_list[-log_interval:])
            print(
                f"  [{pct:5.1f}%] {frame_idx}/{TOTAL_FRAMES}  |  "
                f"avg {avg_ms:.1f} ms/frame  |  경과 {elapsed:.0f}s"
            )

    cap.release()
    writer.release()

    total_elapsed = time.perf_counter() - start_wall
    smoke_ratio   = smoke_frames / max(frame_idx, 1) * 100

    print("\n" + "=" * 55)
    print(f"  처리 프레임   : {frame_idx}장")
    print(f"  총 소요 시간  : {total_elapsed:.1f}초")
    print(f"  평균 속도     : {np.mean(ms_list):.1f} ms/frame  ({1000/np.mean(ms_list):.1f} FPS)")
    print(f"  최종 smoke    : {smoke_frames}장 ({smoke_ratio:.1f}%)")
    print(f"  최종 no_smoke : {frame_idx - smoke_frames}장 ({100-smoke_ratio:.1f}%)")
    print(f"  저장 경로     : {OUT_VIDEO}")
    print("=" * 55)

    # 타임라인 차트 저장
    plot_timeline(frame_results, FPS, TOTAL_FRAMES)

    # 프레임별 결과 CSV 저장
    csv_path = OUT_DIR / "frame_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame_idx", "final_label", "yolo_conf"])
        writer_csv.writerows(frame_results)
    print(f"프레임 결과 저장: {csv_path}")


# ────────────────────────────────────────────────────────────
# 8. 타임라인 시각화
# ────────────────────────────────────────────────────────────

def plot_timeline(frame_results: list, fps: float, total_frames: int):
    """5초 단위 smoke 탐지 비율 타임라인 차트를 저장한다."""
    bin_sec  = 5
    duration = total_frames / fps
    n_bins   = int(np.ceil(duration / bin_sec))
    smoke_per_bin = np.zeros(n_bins)
    total_per_bin = np.zeros(n_bins)

    for fidx, label, _ in frame_results:
        sec     = fidx / fps
        bin_idx = min(int(sec // bin_sec), n_bins - 1)
        total_per_bin[bin_idx] += 1
        if label in ("smoke", "smoke_warning"):
            smoke_per_bin[bin_idx] += 1

    ratio    = smoke_per_bin / np.maximum(total_per_bin, 1)
    x_labels = [f"{i * bin_sec // 60}:{i * bin_sec % 60:02d}" for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(16, 4))
    colors  = ["tomato" if r >= 0.5 else "mediumseagreen" for r in ratio]
    ax.bar(range(n_bins), ratio, color=colors, width=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xticks(range(0, n_bins, max(1, n_bins // 20)))
    ax.set_xticklabels(x_labels[::max(1, n_bins // 20)], rotation=45, ha="right")
    ax.set_xlabel("Time (min:sec)")
    ax.set_ylabel("Smoke Ratio")
    ax.set_title("Smoke Detection Ratio per 5s  (4ch + Hybrid Decision)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "smoke_timeline.png"), dpi=150)
    plt.close()
    print(f"타임라인 저장: {OUT_DIR / 'smoke_timeline.png'}")


if __name__ == "__main__":
    main()
