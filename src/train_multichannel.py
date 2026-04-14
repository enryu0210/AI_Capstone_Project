"""
train_multichannel.py
======================
YOLOv8n-cls 첫 번째 Conv 레이어를 3채널 → 4채널로 수정하고
MultiChannelSmokeDataset으로 직접 PyTorch 학습 루프를 실행한다.

핵심 변경:
  - model.model[0].conv: Conv2d(3 → 4, ...)
  - 4번째 채널 가중치: 기존 3채널 가중치의 평균으로 초기화
    → 랜덤 초기화 대비 학습 초기 안정성 높음
  - YOLO 내장 train() 대신 AdamW + CosineAnnealingLR 직접 사용
    (YOLO 내장 DataLoader가 3채널 전용이라 사용 불가)

출력:
  runs/smoke_detector/multichannel/
    best.pt           ← 최고 Recall 기준 가중치
    last.pt           ← 마지막 epoch 가중치
    training_log.csv  ← epoch별 loss, precision, recall, f1
    pr_curve.png      ← Precision-Recall 곡선 (기존 3채널과 비교 기준)
"""

import csv
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경 대응 (서버/노트북 외부 실행 시)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "smoke_cls_4ch"
OUT_DIR  = BASE_DIR / "runs" / "smoke_detector" / "multichannel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# multichannel_dataset.py 임포트 (같은 src/ 폴더)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multichannel_dataset import MultiChannelSmokeDataset

# ── 하이퍼파라미터 ────────────────────────────────────────
EPOCHS   = 50
IMGSZ    = 224
BATCH    = 32
LR       = 1e-3
PATIENCE = 10
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────────────────
# 1. 모델 아키텍처 구성
# ────────────────────────────────────────────────────────────

def build_4ch_model() -> nn.Module:
    """
    YOLOv8n-cls pretrained 모델의 첫 번째 Conv 레이어를
    3채널 → 4채널로 수정하여 반환한다.

    수정 방식:
      - 새 Conv2d(in_channels=4, ...) 생성
      - ch 0~2 가중치: pretrained 값 그대로 복사
      - ch 3 가중치 : 기존 3채널 평균으로 초기화
        (diff_map이 연기 텍스처와 상관관계가 있으므로 평균으로 시작하는 것이 합리적)

    Returns:
        ClassificationModel (nn.Module) — 4채널 입력을 받는 YOLO 분류 모델
    """
    yolo     = YOLO("yolov8n-cls.pt")
    nn_model = yolo.model  # ClassificationModel (nn.Module)

    # 첫 번째 Conv 레이어 접근
    # 구조: ClassificationModel → .model (nn.Sequential) → [0] (Conv 블록) → .conv (nn.Conv2d)
    first_conv = nn_model.model[0].conv
    print(f"  원본 첫 번째 conv : {first_conv}")

    # 새 Conv2d 생성 (in_channels: 3 → 4, 나머지는 동일)
    new_conv = nn.Conv2d(
        in_channels  = 4,
        out_channels = first_conv.out_channels,
        kernel_size  = first_conv.kernel_size,
        stride       = first_conv.stride,
        padding      = first_conv.padding,
        bias         = first_conv.bias is not None,
    )

    with torch.no_grad():
        # ch 0~2: pretrained 가중치 복사
        new_conv.weight[:, :3, :, :] = first_conv.weight.data.clone()
        # ch 3: 기존 3채널 평균으로 초기화
        new_conv.weight[:, 3:, :, :] = first_conv.weight.data.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    # 교체
    nn_model.model[0].conv = new_conv
    print(f"  수정된 첫 번째 conv: {nn_model.model[0].conv}")

    return nn_model


# ────────────────────────────────────────────────────────────
# 2. 학습 루프
# ────────────────────────────────────────────────────────────

def train():
    print(f"디바이스 : {DEVICE}")
    if "cuda" in DEVICE:
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    # 데이터 로더 구성
    train_ds = MultiChannelSmokeDataset(DATA_DIR, split="train", imgsz=IMGSZ, augment=True)
    val_ds   = MultiChannelSmokeDataset(DATA_DIR, split="val",   imgsz=IMGSZ, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    # 모델, 손실함수, 옵티마이저, 스케줄러
    print("\n모델 구성 중 ...")
    model     = build_4ch_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # T_max=EPOCHS → 학습 끝까지 코사인 스케줄로 lr 감소 (eta_min=LR*0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_recall  = 0.0
    patience_cnt = 0
    log_rows     = []

    print(f"\n학습 시작: {EPOCHS} epochs, batch={BATCH}, lr={LR}\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0         = time.perf_counter()

        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # ClassificationModel forward → (batch, num_classes) logits
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_train_loss = train_loss / len(train_ds)

        # ── Val ───────────────────────────────────────────────
        model.eval()
        tp = fp = fn = tn = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs      = imgs.to(DEVICE)
                preds     = model(imgs).argmax(dim=1).cpu().numpy()
                labels_np = labels.numpy()

                for pred, gt in zip(preds, labels_np):
                    if   gt == 1 and pred == 1: tp += 1   # TP: smoke 정탐
                    elif gt == 1 and pred == 0: fn += 1   # FN: smoke 누락 (가장 치명적)
                    elif gt == 0 and pred == 1: fp += 1   # FP: 오탐
                    else:                       tn += 1   # TN: no_smoke 정탐

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        elapsed   = time.perf_counter() - t0

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] "
            f"loss={avg_train_loss:.4f}  "
            f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  "
            f"TP={tp} FN={fn} FP={fp} TN={tn}  "
            f"({elapsed:.1f}s)"
        )

        log_rows.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

        # ── Best 모델 저장 (Recall 기준) ─────────────────────
        # 연기 누락(FN)이 가장 치명적이므로 Recall을 우선 지표로 사용
        if recall > best_recall:
            best_recall  = recall
            patience_cnt = 0
            torch.save(model.state_dict(), str(OUT_DIR / "best.pt"))
            print(f"  → best.pt 저장 (Recall={recall:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[Early Stop] {PATIENCE} epochs 동안 Recall 개선 없음. 종료.")
                break

    # 마지막 epoch 가중치 저장 (비교/복구용)
    torch.save(model.state_dict(), str(OUT_DIR / "last.pt"))

    # ── CSV 학습 로그 저장 ────────────────────────────────────
    csv_path = OUT_DIR / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nCSV 로그 저장: {csv_path}")

    # ── PR 분석 ───────────────────────────────────────────────
    print("\nPR 분석 시작 (val 세트) ...")
    pr_analysis(model, val_loader)

    print(f"\n[결과] Best Recall = {best_recall:.4f}")
    print(f"[결과] 가중치 저장: {OUT_DIR / 'best.pt'}")

    # ── 기존 3채널 모델과 비교 ────────────────────────────────
    print("\n[비교] 기존 3채널 모델 (threshold=0.40 기준)")
    print("       Recall=0.9635, Precision=0.9788, F1=0.9711, FN=7")
    print(f"[비교] 4채널 모델 Best Recall={best_recall:.4f}")


# ────────────────────────────────────────────────────────────
# 3. Precision-Recall 분석
# ────────────────────────────────────────────────────────────

def pr_analysis(model: nn.Module, val_loader: DataLoader):
    """
    val 세트의 smoke softmax 확률을 수집하고,
    threshold 0.05~0.95 구간별 Precision/Recall/F1을 측정한다.
    기존 3채널 모델과 동일한 방식으로 분석하여 직접 비교 가능하게 함.
    """
    model.eval()
    smoke_confs    = []  # GT=smoke 샘플의 smoke 확률
    no_smoke_confs = []  # GT=no_smoke 샘플의 smoke 확률

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()  # (batch, 2)

            for i, label in enumerate(labels.numpy()):
                smoke_prob = float(probs[i, 1])  # smoke 클래스 index=1
                if label == 1:
                    smoke_confs.append(smoke_prob)
                else:
                    no_smoke_confs.append(smoke_prob)

    print(f"smoke    conf → mean={np.mean(smoke_confs):.3f}  min={np.min(smoke_confs):.3f}")
    print(f"no_smoke conf → mean={np.mean(no_smoke_confs):.3f}  max={np.max(no_smoke_confs):.3f}")

    thresholds           = [i / 20 for i in range(1, 20)]
    precisions, recalls, f1s = [], [], []

    print(f"\n{'threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | TP | FP | FN | TN")
    print("-" * 70)

    best_f1, best_t = 0, 0
    for t in thresholds:
        tp = sum(1 for c in smoke_confs    if c >= t)
        fn = sum(1 for c in smoke_confs    if c <  t)
        fp = sum(1 for c in no_smoke_confs if c >= t)
        tn = sum(1 for c in no_smoke_confs if c <  t)
        prec   = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1     = 2 * prec * recall / max(prec + recall, 1e-9)
        precisions.append(prec); recalls.append(recall); f1s.append(f1)

        marker = " ← best F1" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
        print(
            f"  {t:8.2f}   | {prec:10.4f} | {recall:8.4f} | {f1:8.4f} | "
            f"{tp:3d}| {fp:3d}| {fn:3d}| {tn:3d}{marker}"
        )

    print(f"\n최적 F1 threshold = {best_t:.2f}  (F1={best_f1:.4f})")

    # PR Curve 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(recalls, precisions, "b-o", markersize=4, label="4ch 모델")
    # 기존 3채널 단일 포인트도 함께 표시 (threshold=0.40 기준)
    axes[0].plot(0.9635, 0.9788, "r*", markersize=12, label="3ch (t=0.40, 기준)")
    for i, t in enumerate(thresholds):
        if t in (0.30, 0.50, best_t):
            axes[0].annotate(f"t={t:.2f}", (recalls[i], precisions[i]),
                             textcoords="offset points", xytext=(5, -10), fontsize=8)
    axes[0].axvline(x=0.99, color="red", linestyle="--", alpha=0.5, label="Recall 목표 0.99")
    axes[0].set_xlabel("Recall (smoke)"); axes[0].set_ylabel("Precision (smoke)")
    axes[0].set_title("Precision-Recall Curve (4ch vs 3ch)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(thresholds, precisions, "b-o", markersize=4, label="Precision")
    axes[1].plot(thresholds, recalls,    "r-o", markersize=4, label="Recall")
    axes[1].plot(thresholds, f1s,        "g-o", markersize=4, label="F1")
    axes[1].axvline(x=best_t, color="gray", linestyle="--", label=f"Best F1 (t={best_t:.2f})")
    axes[1].set_xlabel("Confidence Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("Threshold vs Metrics (4ch)")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "pr_curve.png"), dpi=150)
    plt.close()
    print(f"PR curve 저장: {OUT_DIR / 'pr_curve.png'}")


# ────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"[오류] 4채널 데이터셋이 없습니다: {DATA_DIR}")
        print("먼저 prepare_multichannel_dataset.py 를 실행하세요.")
        sys.exit(1)
    train()
