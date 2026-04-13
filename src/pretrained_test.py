"""
pretrained_test.py
==================
fine-tuning 전, ImageNet pretrained YOLOv8n-cls 가중치로
수술 영상 샘플을 추론하여 기준선(baseline) 성능을 측정한다.

- 모델  : yolov8n-cls.pt (ultralytics 자동 다운로드)
- 샘플  : val/smoke 50장, val/no_smoke 50장
- 지표  : Accuracy, Recall(smoke), Precision(smoke), F1(smoke)
- conf  : 0.3 (연기를 놓치는 위험 최소화)
- 목표  : Recall ≥ 0.99 → 충족 시 fine-tuning 불필요
"""

import random
import time
from pathlib import Path

from ultralytics import YOLO


# ── 설정 ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
VAL_DIR    = BASE_DIR / "data" / "smoke_cls" / "val"
SAMPLE_N   = 50          # 클래스당 샘플 수
CONF_THRES = 0.3         # smoke 판정 최소 confidence (낮을수록 recall↑)
RANDOM_SEED = 42
TARGET_RECALL = 0.99


def sample_images(class_dir: Path, n: int) -> list[Path]:
    """클래스 폴더에서 n장을 랜덤 샘플링한다."""
    all_imgs = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    if len(all_imgs) < n:
        print(f"  [경고] {class_dir.name} 이미지가 {len(all_imgs)}장뿐 → 전체 사용")
        return all_imgs
    return random.sample(all_imgs, n)


def predict_class(model: YOLO, img_path: Path) -> tuple[str, float]:
    """
    단일 이미지 추론.
    반환: (예측 클래스명, smoke confidence)
    YOLO cls 모델은 results[0].probs 에 클래스별 확률을 반환한다.
    """
    results = model.predict(str(img_path), verbose=False)
    probs   = results[0].probs          # ultralytics Probs 객체
    names   = results[0].names          # {0: 'no_smoke', 1: 'smoke'} 등

    # smoke 클래스 인덱스 탐색 (학습 전이므로 ImageNet 클래스명과 다를 수 있음)
    # pretrained 모델은 ImageNet 1000개 클래스 → smoke/no_smoke 개념 없음
    # top1 클래스 index와 confidence를 그대로 사용하여 임계값으로 판단
    top1_idx  = int(probs.top1)
    top1_conf = float(probs.top1conf)

    # pretrained 모델에서 'smoke' 감지 대리 지표:
    # conf가 CONF_THRES 미만이면 "모르겠다" → 안전하게 smoke로 판정
    # → False Negative 최소화 전략
    if top1_conf < CONF_THRES:
        pred_label = "smoke"
    else:
        # ImageNet 클래스명에 smoke 관련 키워드가 있으면 smoke로 판정
        cls_name = names[top1_idx].lower()
        smoke_keywords = ("smoke", "fog", "haze", "mist", "steam", "fire")
        pred_label = "smoke" if any(k in cls_name for k in smoke_keywords) else "no_smoke"

    return pred_label, top1_conf


def calc_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """이진 분류 지표 계산 (smoke = positive)."""
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def main():
    random.seed(RANDOM_SEED)

    # ── 1. 모델 로드 ──────────────────────────────────────
    print("[로드] yolov8n-cls.pt (ImageNet pretrained)")
    model = YOLO("yolov8n-cls.pt")   # 없으면 ultralytics가 자동 다운로드

    # ── 2. 샘플 수집 ──────────────────────────────────────
    smoke_imgs    = sample_images(VAL_DIR / "smoke",    SAMPLE_N)
    no_smoke_imgs = sample_images(VAL_DIR / "no_smoke", SAMPLE_N)
    print(f"[샘플] smoke {len(smoke_imgs)}장 / no_smoke {len(no_smoke_imgs)}장")

    # ── 3. 추론 ───────────────────────────────────────────
    tp = fp = fn = tn = 0
    total_ms = 0.0

    print("\n[추론] smoke 샘플...")
    for img in smoke_imgs:
        t0 = time.perf_counter()
        pred, _ = predict_class(model, img)
        total_ms += (time.perf_counter() - t0) * 1000
        if pred == "smoke":
            tp += 1   # 정답: smoke, 예측: smoke
        else:
            fn += 1   # 정답: smoke, 예측: no_smoke (가장 위험한 오류)

    print("[추론] no_smoke 샘플...")
    for img in no_smoke_imgs:
        t0 = time.perf_counter()
        pred, _ = predict_class(model, img)
        total_ms += (time.perf_counter() - t0) * 1000
        if pred == "no_smoke":
            tn += 1   # 정답: no_smoke, 예측: no_smoke
        else:
            fp += 1   # 정답: no_smoke, 예측: smoke

    # ── 4. 지표 계산 ──────────────────────────────────────
    metrics = calc_metrics(tp, fp, fn, tn)
    avg_ms  = total_ms / (len(smoke_imgs) + len(no_smoke_imgs))

    # ── 5. 결과 출력 ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Pretrained YOLOv8n-cls 추론 결과 (Baseline)")
    print("=" * 50)
    print(f"  conf threshold : {CONF_THRES}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall (smoke) : {metrics['recall']:.4f}  ← 핵심 지표")
    print(f"  F1 Score       : {metrics['f1']:.4f}")
    print(f"  평균 추론 속도 : {avg_ms:.1f} ms/프레임")
    print("=" * 50)

    # ── 6. fine-tuning 필요 여부 판단 ────────────────────
    if metrics['recall'] >= TARGET_RECALL:
        print(f"\n[판단] Recall {metrics['recall']:.4f} ≥ {TARGET_RECALL}"
              f" → fine-tuning 불필요 ✓")
    else:
        gap = TARGET_RECALL - metrics['recall']
        fn_count = fn
        print(f"\n[판단] Recall {metrics['recall']:.4f} < {TARGET_RECALL}"
              f" (부족분 {gap:.4f})")
        print(f"        FN(연기 놓침) {fn_count}건 → 단계 3 fine-tuning 진행 필요")


if __name__ == "__main__":
    main()
