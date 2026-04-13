"""
prepare_dataset.py
==================
DesmokeData_dataset → YOLO classification 형식으로 재구성

원본 구조 (단일 플랫 폴더):
  assets/DesmokeData_dataset/
    1.png        ← smoke 이미지 (gt 없음)
    1_gt.png     ← no_smoke 이미지 (gt 붙음)

출력 구조:
  data/smoke_cls/
    train/ smoke/ & no_smoke/
    val/   smoke/ & no_smoke/

핵심 변경: 복사 시 CLAHE 전처리 적용
  - 조명 환경이 달라도 동일한 대비 정규화를 거치므로 일반화 성능 향상
  - 학습/추론 모두 동일한 CLAHE를 적용해야 일관성 유지
"""

import random
import shutil
import cv2
import numpy as np
from pathlib import Path


# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
SRC_DIR    = BASE_DIR / "assets" / "DesmokeData_dataset"
OUTPUT_DIR = BASE_DIR / "data" / "smoke_cls"

RANDOM_SEED = 42
VAL_RATIO   = 0.2

# CLAHE 설정
# clipLimit: 대비 강조 한계 (높을수록 강하게 정규화)
# tileGridSize: 지역 단위 크기 (8x8이 표준)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR 이미지에 CLAHE 적용.
    LAB 색공간의 L(밝기) 채널에만 적용해 색감은 유지하면서 대비만 정규화.
    → 핀포인트 조명처럼 국소적으로 밝기 편차가 큰 경우 효과적.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = CLAHE.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def copy_with_clahe(src: Path, dst: Path):
    """원본을 CLAHE 정규화 후 저장. 실패 시 원본 그대로 복사."""
    img = cv2.imread(str(src))
    if img is None:
        shutil.copy(src, dst)
        return
    processed = apply_clahe(img)
    cv2.imwrite(str(dst), processed)


def split_and_copy(src_paths: list[Path], train_dst: Path, val_dst: Path) -> tuple[int, int]:
    """src_paths를 8:2로 섞어 분할 후 CLAHE 적용해 저장."""
    random.shuffle(src_paths)
    n_val   = int(len(src_paths) * VAL_RATIO)
    val_set = src_paths[:n_val]
    trn_set = src_paths[n_val:]

    for dst_dir, file_list in [(train_dst, trn_set), (val_dst, val_set)]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in file_list:
            copy_with_clahe(src, dst_dir / src.name)

    return len(trn_set), len(val_set)


def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"원본 데이터셋 폴더를 찾을 수 없음: {SRC_DIR}")

    if OUTPUT_DIR.exists():
        print(f"[초기화] 기존 폴더 삭제: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    all_files = list(SRC_DIR.glob("*.png")) + list(SRC_DIR.glob("*.jpg"))
    smoke_paths    = [f for f in all_files if "_gt" not in f.stem]
    no_smoke_paths = [f for f in all_files if "_gt" in f.stem]

    print(f"[수집] smoke(gt 없음) : {len(smoke_paths)}장")
    print(f"[수집] no_smoke(gt)   : {len(no_smoke_paths)}장")
    print(f"[전처리] CLAHE 적용 (clipLimit=2.0, tile=8×8) 후 저장")

    if not smoke_paths or not no_smoke_paths:
        raise ValueError("smoke 또는 no_smoke 이미지가 0장입니다.")

    random.seed(RANDOM_SEED)

    print("\n[복사] smoke → train/smoke & val/smoke")
    trn_s, val_s = split_and_copy(
        smoke_paths,
        train_dst = OUTPUT_DIR / "train" / "smoke",
        val_dst   = OUTPUT_DIR / "val"   / "smoke",
    )

    print("[복사] no_smoke → train/no_smoke & val/no_smoke")
    trn_n, val_n = split_and_copy(
        no_smoke_paths,
        train_dst = OUTPUT_DIR / "train" / "no_smoke",
        val_dst   = OUTPUT_DIR / "val"   / "no_smoke",
    )

    print("\n[완료] 데이터셋 재구성 결과")
    print("=" * 45)
    print(f"  train/smoke      : {trn_s:5d}장")
    print(f"  train/no_smoke   : {trn_n:5d}장")
    print(f"  val  /smoke      : {val_s:5d}장")
    print(f"  val  /no_smoke   : {val_n:5d}장")
    print(f"  합계             : {trn_s + trn_n + val_s + val_n:5d}장")
    print("=" * 45)

    ratio = max(trn_s, trn_n) / max(min(trn_s, trn_n), 1)
    if ratio > 1.5:
        print(f"\n[경고] 클래스 불균형 감지 (비율 {ratio:.2f}x)")
    else:
        print(f"\n[정상] 클래스 균형 양호 (비율 {ratio:.2f}x)")
    print(f"\n출력 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
