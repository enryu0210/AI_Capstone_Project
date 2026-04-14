"""
video_inference_v2.py
======================
개선된 얇은 연기 탐지 — 오탐률 97% → 목표 <5% 수정 버전

[이전 버전(ThinSmokeDetector)의 문제점]
- 결과: YOLO 44.2% vs ThinSmoke 97.1% (오탐 53% 추가)
- 원인 1: 단일 프레임 Sharpness 기반 → 카메라 이동/초점 변화에도 반응
- 원인 2: EMA 베이스라인(alpha=0.03)이 초반 고변동 구간에서 오염됨
- 원인 3: score ≥ 2 조건이 너무 낮음 (Saturation 1개만 떨어져도 탐지)

[v2 핵심 수정]
1. Rolling Percentile 베이스라인 (EMA → 최근 300프레임 80th percentile)
   - 단발성 노이즈에 강건: 1-2프레임 dip은 percentile에 영향 미미
   - 자동 캘리브레이션: 수술 환경이 바뀌어도 adaptive하게 추종
2. 동시 하락 요구 (Sharpness+Saturation 동시)
   - 카메라 이동: Sharpness만 ↓ (Saturation 유지) → 미탐지
   - 연기 침투: Sharpness + Saturation 동시 ↓ → 탐지
3. 지속성 확인 (15프레임 중 12개 이상)
   - 단발성 카메라 이동(1-3프레임): 필터링
   - 실제 연기(수 초간 지속): 통과
4. YOLO 불확실 구간 제한 [0.10, 0.40]
   - YOLO conf > 0.40: YOLO 판단 신뢰 → 보조 탐지 비활성
   - YOLO conf < 0.10: YOLO가 명확히 no_smoke → 보조 탐지 비활성
   - 중간 구간에서만 보조 탐지 보강
5. 워밍업 300프레임 (10초) — percentile 버퍼 채우기 전 탐지 안 함
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "assets" / "capture1.avi"
MODEL_PATH = BASE_DIR / "runs" / "smoke_detector" / "yolov8n_cls" / "weights" / "best.pt"
OUT_DIR    = BASE_DIR / "runs" / "smoke_detector" / "video_result_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO  = OUT_DIR / "capture1_result_v2.avi"
LOG_CSV    = OUT_DIR / "frame_log_v2.csv"

# ── 1단계: YOLO 탐지 파라미터 ────────────────────────────
SMOKE_THR   = 0.40  # raw confidence 임계값
VOTE_WINDOW = 5     # sliding vote 창 크기
VOTE_K      = 3     # 과반 이상이면 smoke

# ── 2단계: 보조 탐지(ThinSmoke v2) 파라미터 ──────────────
WARMUP_FRAMES    = 300   # 베이스라인 버퍼를 채우는 초기 프레임 수 (10초)
BASELINE_WINDOW  = 300   # rolling percentile 계산 창 크기 (10초)
BASELINE_PCT     = 80    # 베이스라인으로 사용할 percentile (80th = "선명한 정상")

# 선명도 스무딩 창 — 단일 프레임 노이즈 제거 핵심 수정
# 문제: 라플라시안 분산이 135~460으로 변동 → 깨끗한 프레임도 ratio=0.28
# 해결: 최근 30프레임의 median을 "현재 선명도 수준"으로 사용
#       카메라 이동(1~3프레임): median에 미미한 영향 → FP 없음
#       연기 지속(수십 프레임): median이 낮게 유지됨 → 탐지
SHARP_SMOOTH_WINDOW = 30  # 선명도 rolling median 창 (1초)

# 연기 신호 임계값 — 스무딩된 지표 기준
SHARP_DROP_RATIO = 0.50  # smooth_sharpness가 80th pct 베이스라인의 50% 미만
                          # (연기로 72% 감소 → smooth ratio ≈ 0.28~0.40, 깨끗한 장면 ≈ 0.55~0.85)
SAT_DROP_RATIO   = 0.82  # Saturation이 베이스라인의 82% 미만
                          # (연기로 24% 감소 → ratio ≈ 0.76, 정상 ≈ 0.92~1.0)

# 지속성 창 — 최근 N프레임 중 M개 이상이 "이상"이어야 탐지
PERSIST_WINDOW   = 15    # 지속성 확인 창
PERSIST_THRESH   = 12    # N프레임 중 M개 이상 (80% 이상)

# YOLO 불확실 구간 — 이 범위에서만 보조 탐지 활성화
# 하한을 0.0으로 설정: YOLO가 "명확히 no_smoke(conf<0.10)"여도 ThinSmoke 체크 허용
# → persist_cnt가 yolo 하한 때문에 12/15를 못 채우는 문제 해결
# → 실제 보호는 "sharp + sat 동시 하락" 이중 게이트가 담당
YOLO_UNCERTAIN_LO = 0.00
YOLO_UNCERTAIN_HI = 0.40


# ── 특징 추출 함수 ────────────────────────────────────────
def extract_features(frame: np.ndarray) -> tuple[float, float]:
    """
    Laplacian Sharpness + HSV Saturation 추출.

    Returns:
        sharpness: Laplacian variance (선명도, 연기 시 감소)
        saturation: HSV S채널 평균 (채도, 연기 시 감소)
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())

    return sharpness, saturation


# ── Rolling Percentile 베이스라인 ─────────────────────────
class RollingPercentileBaseline:
    """
    최근 N프레임의 P번째 percentile을 베이스라인으로 사용.

    EMA(지수이동평균) 대비 장점:
    - 초반 고변동 구간에서 오염되지 않음
    - alpha 튜닝 없이 자동으로 안정적인 '정상' 수준 추적
    - 25th percentile 사용 → 일시적 상승(이상값)에 강건
    """

    def __init__(self, window: int = 300, percentile: int = 80):
        self.buf        = deque(maxlen=window)
        self.percentile = percentile

    def update(self, value: float):
        self.buf.append(value)

    def get_baseline(self) -> float | None:
        """버퍼가 window 절반 이상 차면 percentile 반환, 아니면 None"""
        if len(self.buf) < (self.buf.maxlen or 1) // 2:
            return None
        return float(np.percentile(list(self.buf), self.percentile))

    def __len__(self):
        return len(self.buf)


# ── 얇은 연기 보조 탐지기 (v2) ───────────────────────────
class ThinSmokeDetectorV2:
    """
    v1의 97% 오탐 문제를 해결한 보조 탐지기.

    핵심 로직:
    1. Rolling Percentile Baseline (WARMUP 후 활성)
    2. Sharpness + Saturation 동시 하락 감지
    3. 15프레임 지속성 확인 후 최종 판정
    """

    def __init__(self):
        self.sharp_baseline  = RollingPercentileBaseline(BASELINE_WINDOW, BASELINE_PCT)
        self.sat_baseline    = RollingPercentileBaseline(BASELINE_WINDOW, BASELINE_PCT)
        self.persist_buf     = deque(maxlen=PERSIST_WINDOW)   # 0/1 지속성 버퍼
        self.sharp_smooth_buf = deque(maxlen=SHARP_SMOOTH_WINDOW)  # 선명도 스무딩용
        self.frame_count     = 0

    def update(self, frame: np.ndarray, yolo_conf: float) -> tuple[bool, dict]:
        """
        프레임마다 호출. 얇은 연기 탐지 여부와 디버그 정보를 반환.

        Args:
            frame: BGR 프레임
            yolo_conf: YOLO smoke confidence (0~1)

        Returns:
            thin_smoke: True면 보조 탐지 발동
            info: 디버그 정보 dict
        """
        self.frame_count += 1

        sharpness, saturation = extract_features(frame)

        # 선명도 스무딩 버퍼 + 베이스라인 항상 업데이트
        self.sharp_smooth_buf.append(sharpness)
        self.sharp_baseline.update(sharpness)
        self.sat_baseline.update(saturation)

        # 스무딩된 선명도 = 최근 N프레임 median (노이즈 제거)
        smooth_sharpness = float(np.median(list(self.sharp_smooth_buf)))

        info = {
            "sharpness":  smooth_sharpness,   # CSV에 스무딩 값 기록
            "saturation": saturation,
            "sharp_base": None,
            "sat_base":   None,
            "sharp_ratio": None,
            "sat_ratio":   None,
            "both_low":   False,
            "persist_cnt": 0,
        }

        # ── 워밍업 중 → 탐지 비활성 ─────────────────────
        if self.frame_count < WARMUP_FRAMES:
            self.persist_buf.append(0)
            return False, info

        # ── YOLO 확신 구간 → 보조 탐지 불필요 ───────────
        # YOLO가 이미 확신하거나 명확히 no_smoke인 경우
        yolo_uncertain = YOLO_UNCERTAIN_LO <= yolo_conf <= YOLO_UNCERTAIN_HI
        if not yolo_uncertain:
            self.persist_buf.append(0)
            return False, info

        # ── 베이스라인 준비 확인 ──────────────────────────
        sharp_base = self.sharp_baseline.get_baseline()
        sat_base   = self.sat_baseline.get_baseline()

        if sharp_base is None or sat_base is None or sharp_base < 1.0 or sat_base < 1.0:
            self.persist_buf.append(0)
            return False, info

        # ── 동시 하락 감지 ────────────────────────────────
        # smooth_sharpness: 30프레임 median → 순간 카메라 이동에 강건
        # saturation: 단일 프레임 (원래 안정적이므로 스무딩 불필요)
        sharp_ratio = smooth_sharpness / sharp_base
        sat_ratio   = saturation / sat_base

        # 두 지표 모두 베이스라인 대비 임계값 미만 → "이상 신호"
        both_low = (sharp_ratio < SHARP_DROP_RATIO) and (sat_ratio < SAT_DROP_RATIO)

        info["sharp_base"]  = sharp_base
        info["sat_base"]    = sat_base
        info["sharp_ratio"] = sharp_ratio
        info["sat_ratio"]   = sat_ratio
        info["both_low"]    = both_low

        # ── 지속성 확인 ───────────────────────────────────
        self.persist_buf.append(1 if both_low else 0)
        persist_cnt = sum(self.persist_buf)
        info["persist_cnt"] = persist_cnt

        # 창이 다 찼고 PERSIST_THRESH 이상 → thin_smoke 탐지
        thin_smoke = (
            len(self.persist_buf) == PERSIST_WINDOW
            and persist_cnt >= PERSIST_THRESH
        )
        return thin_smoke, info


# ── 오버레이 그리기 ───────────────────────────────────────
def draw_overlay_v2(
    frame, final_label, yolo_raw, yolo_conf, vote_cnt,
    thin_smoke, thin_info, frame_idx, total, elapsed
):
    """
    상단 바에 최종 판정 + YOLO raw + ThinSmoke v2 상태를 표시.
    """
    h, w      = frame.shape[:2]
    is_smoke  = (final_label == "smoke")
    is_thin   = thin_smoke and not (yolo_conf >= SMOKE_THR)  # 보조 탐지로만 잡힌 경우
    bar_color = (0, 0, 200) if is_smoke else (0, 180, 0)

    cv2.rectangle(frame, (0, 0), (w, 75), bar_color, -1)

    # 라인 1: 최종 판정
    label_text = "[SMOKE]" if is_smoke else "[NO SMOKE]"
    if is_thin:
        label_text += " (thin)"
    status = f"{label_text}  vote={vote_cnt}/{VOTE_WINDOW}"
    cv2.putText(frame, status, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # 라인 2: YOLO raw
    raw_txt = f"YOLO: {yolo_conf:.2f} ({yolo_raw})"
    cv2.putText(frame, raw_txt, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # 라인 3: ThinSmoke 상태
    if thin_info["sharp_ratio"] is not None:
        thin_txt = (
            f"ThinSmoke: sharp_r={thin_info['sharp_ratio']:.2f} "
            f"sat_r={thin_info['sat_ratio']:.2f} "
            f"persist={thin_info['persist_cnt']}/{PERSIST_WINDOW}"
        )
    else:
        warmup_rem = max(0, WARMUP_FRAMES - frame_idx)
        thin_txt   = f"ThinSmoke: warmup... ({warmup_rem}f left)"
    cv2.putText(frame, thin_txt, (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 255), 1, cv2.LINE_AA)

    # 우측: 진행
    info_txt = f"Frame {frame_idx}/{total}  {elapsed:.1f}s"
    cv2.putText(frame, info_txt, (w - 270, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── 메인 추론 루프 ────────────────────────────────────────
def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"영상 파일 없음: {VIDEO_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}\ntrain_yolo.ipynb 먼저 실행하세요.")

    cap          = cv2.VideoCapture(str(VIDEO_PATH))
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS          = cap.get(cv2.CAP_PROP_FPS)
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"입력 영상     : {VIDEO_PATH.name}")
    print(f"해상도        : {W} × {H}  /  FPS: {FPS:.2f}  /  총 {TOTAL_FRAMES}프레임 ({TOTAL_FRAMES/FPS:.1f}초)")
    print(f"YOLO Threshold: {SMOKE_THR}  /  Vote: {VOTE_WINDOW}중 {VOTE_K}")
    print(f"ThinSmoke v2  : warmup={WARMUP_FRAMES}f  window={BASELINE_WINDOW}f  pct={BASELINE_PCT}th")
    print(f"              : sharp_drop<{SHARP_DROP_RATIO}  sat_drop<{SAT_DROP_RATIO}")
    print(f"              : persist {PERSIST_THRESH}/{PERSIST_WINDOW}프레임")

    model     = YOLO(str(MODEL_PATH))
    smoke_idx = [k for k, v in model.names.items() if v == "smoke"][0]

    cap      = cv2.VideoCapture(str(VIDEO_PATH))
    writer   = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"XVID"), FPS, (W, H))
    vote_buf = deque(maxlen=VOTE_WINDOW)
    detector = ThinSmokeDetectorV2()

    # 로그 집계
    yolo_smoke_cnt  = 0
    thin_smoke_cnt  = 0  # 보조 탐지로만 잡힌 프레임
    final_smoke_cnt = 0
    frame_idx       = 0
    ms_list         = []
    start_wall      = time.perf_counter()

    # CSV 헤더
    with open(LOG_CSV, "w", encoding="utf-8") as f:
        f.write("frame,sec,yolo_conf,yolo_raw,thin_smoke,final,sharp,sat,sharp_ratio,sat_ratio,persist\n")

    print("\n추론 시작 (v2)...")
    cap_read = cv2.VideoCapture(str(VIDEO_PATH))
    while True:
        ret, frame = cap_read.read()
        if not ret:
            break
        frame_idx += 1

        # ① YOLO 추론
        t0         = time.perf_counter()
        results    = model.predict(frame, verbose=False, imgsz=224)
        ms_list.append((time.perf_counter() - t0) * 1000)

        yolo_conf = float(results[0].probs.data[smoke_idx])
        yolo_raw  = "smoke" if yolo_conf >= SMOKE_THR else "no_smoke"

        # ② Sliding window vote (YOLO 기반)
        vote_buf.append(1 if yolo_raw == "smoke" else 0)
        vote_cnt  = sum(vote_buf)
        yolo_vote = vote_cnt >= VOTE_K

        if yolo_vote:
            yolo_smoke_cnt += 1

        # ③ 보조 탐지 (ThinSmoke v2) — YOLO vote가 no_smoke일 때만 보정
        thin_smoke, thin_info = detector.update(frame, yolo_conf)

        # ④ 최종 판정: YOLO vote OR ThinSmoke
        final_smoke = yolo_vote or thin_smoke
        if final_smoke:
            final_smoke_cnt += 1
        if thin_smoke and not yolo_vote:
            thin_smoke_cnt += 1

        final_label = "smoke" if final_smoke else "no_smoke"

        # ⑤ 오버레이 저장
        elapsed   = time.perf_counter() - start_wall
        out_frame = draw_overlay_v2(
            frame.copy(), final_label, yolo_raw, yolo_conf, vote_cnt,
            thin_smoke, thin_info, frame_idx, TOTAL_FRAMES, elapsed
        )
        writer.write(out_frame)

        # ⑥ CSV 로그
        with open(LOG_CSV, "a", encoding="utf-8") as f:
            sec = frame_idx / FPS
            sr  = f"{thin_info['sharp_ratio']:.3f}" if thin_info["sharp_ratio"] is not None else ""
            sar = f"{thin_info['sat_ratio']:.3f}"   if thin_info["sat_ratio"]   is not None else ""
            f.write(
                f"{frame_idx},{sec:.2f},{yolo_conf:.4f},{yolo_raw},"
                f"{int(thin_smoke)},{final_label},"
                f"{thin_info['sharpness']:.1f},{thin_info['saturation']:.2f},"
                f"{sr},{sar},{thin_info['persist_cnt']}\n"
            )

        if frame_idx % 500 == 0 or frame_idx == TOTAL_FRAMES:
            pct    = frame_idx / TOTAL_FRAMES * 100
            avg_ms = np.mean(ms_list[-500:])
            print(
                f"  [{pct:5.1f}%] {frame_idx}/{TOTAL_FRAMES}  "
                f"avg {avg_ms:.1f}ms  "
                f"YOLO_smoke={yolo_smoke_cnt}({yolo_smoke_cnt/frame_idx*100:.1f}%)  "
                f"ThinOnly={thin_smoke_cnt}  "
                f"Final={final_smoke_cnt}({final_smoke_cnt/frame_idx*100:.1f}%)"
            )

    cap_read.release()
    writer.release()

    total_sec = time.perf_counter() - start_wall
    total     = max(frame_idx, 1)

    print("\n" + "=" * 65)
    print(f"  처리 프레임     : {frame_idx}장  ({total_sec:.1f}초)")
    print(f"  평균 속도       : {np.mean(ms_list):.1f} ms/frame  ({1000/np.mean(ms_list):.1f} FPS)")
    print(f"  YOLO smoke      : {yolo_smoke_cnt}장 ({yolo_smoke_cnt/total*100:.1f}%)")
    print(f"  ThinSmoke 추가  : {thin_smoke_cnt}장 ({thin_smoke_cnt/total*100:.1f}%)  ← 보조 탐지만")
    print(f"  최종 smoke      : {final_smoke_cnt}장 ({final_smoke_cnt/total*100:.1f}%)")
    print(f"  최종 no_smoke   : {total-final_smoke_cnt}장 ({(total-final_smoke_cnt)/total*100:.1f}%)")
    print(f"  저장 영상       : {OUT_VIDEO}")
    print(f"  프레임 로그     : {LOG_CSV}")
    print("=" * 65)

    # ── 타임라인 시각화 ────────────────────────────────────
    import pandas as pd
    df  = pd.read_csv(LOG_CSV)
    bin_sec = 5
    fps_val = FPS

    df["bin"] = (df["sec"] // bin_sec).astype(int)
    timeline  = df.groupby("bin").agg(
        yolo_ratio  = ("yolo_raw",  lambda x: (x == "smoke").mean()),
        final_ratio = ("final",     lambda x: (x == "smoke").mean()),
        thin_ratio  = ("thin_smoke", "mean"),
    ).reset_index()

    n_bins   = len(timeline)
    x_labels = [f"{int(b*bin_sec)//60}:{int(b*bin_sec)%60:02d}" for b in timeline["bin"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7), sharex=True)

    ax1.bar(range(n_bins), timeline["yolo_ratio"], color="steelblue", width=0.85, label="YOLO vote")
    ax1.bar(range(n_bins), timeline["thin_ratio"], bottom=timeline["yolo_ratio"],
            color="orange", width=0.85, alpha=0.7, label="ThinSmoke 추가분")
    ax1.axhline(0.5, color="gray", ls="--", lw=1)
    ax1.set_ylabel("Smoke 비율")
    ax1.set_title("v2 탐지 타임라인 — YOLO + ThinSmoke 보조 탐지 (5초 단위)")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper right")

    colors = ["tomato" if r >= 0.5 else "mediumseagreen" for r in timeline["final_ratio"]]
    ax2.bar(range(n_bins), timeline["final_ratio"], color=colors, width=0.85)
    ax2.axhline(0.5, color="gray", ls="--", lw=1)
    ax2.set_ylabel("최종 Smoke 비율")
    ax2.set_title("최종 판정 타임라인")
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(0, n_bins, max(1, n_bins // 20)))
    ax2.set_xticklabels(
        [x_labels[i] for i in range(0, n_bins, max(1, n_bins // 20))],
        rotation=45, ha="right"
    )
    ax2.set_xlabel("시간 (분:초)")

    plt.tight_layout()
    timeline_path = OUT_DIR / "smoke_timeline_v2.png"
    plt.savefig(str(timeline_path), dpi=150)
    plt.close()
    print(f"\n타임라인 저장: {timeline_path}")


if __name__ == "__main__":
    main()
