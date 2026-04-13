"""
prepare_dataset.py
==================
원본 DesmokeData → YOLO classification 형식으로 재구성

입력 구조:
  assets/dataset/DesmokeData/train/input/   ← smoky 이미지
  assets/dataset/DesmokeData/train/target/  ← clean 이미지
  assets/dataset/DesmokeData/test/input/    ← smoky 이미지
  assets/dataset/DesmokeData/test/target/   ← clean 이미지

출력 구조 (data/smoke_cls/):
  train/smoke/     train/no_smoke/
  val/smoke/       val/no_smoke/

분할 비율: train 80% / val 20%
주의: 원본은 절대 이동하지 않고 shutil.copy 만 사용
"""

import os
import random
import shutil
from pathlib import Path


# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent          # 프로젝트 루트
DATA_ROOT   = BASE_DIR / "assets" / "dataset" / "DesmokeData"
OUTPUT_DIR  = BASE_DIR / "data" / "smoke_cls"

# 원본 split 목록 (train + test 모두 합쳐서 재분할)
SPLITS      = ["train", "test"]

# 재현성을 위한 시드
RANDOM_SEED = 42
VAL_RATIO   = 0.2


def collect_image_paths(class_key: str) -> list[Path]:
    """
    class_key: 'input'(smoke) 또는 'target'(no_smoke)
    train/ + test/ 두 폴더에서 이미지 경로를 수집해 합친다.
    """
    paths = []
    for split in SPLITS:
        folder = DATA_ROOT / split / class_key
        if not folder.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없음: {folder}")
        # PNG·JPG·JPEG 확장자만 수집
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(folder.glob(ext))
    return paths


def split_and_copy(src_paths: list[Path], train_dst: Path, val_dst: Path):
    """
    src_paths를 8:2로 섞어 분할한 뒤 각 목적지 폴더로 복사한다.
    파일명 충돌 방지를 위해 원본 split 이름을 prefix로 추가한다.
    (예: train_cholec_1.png, test_cholec_1.png)
    """
    random.shuffle(src_paths)
    n_val   = int(len(src_paths) * VAL_RATIO)
    val_set = src_paths[:n_val]
    trn_set = src_paths[n_val:]

    for dst_dir, file_list in [(train_dst, trn_set), (val_dst, val_set)]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in file_list:
            # 부모 폴더명(train/test)을 prefix로 붙여 충돌 방지
            prefix   = src.parent.parent.name   # "train" 또는 "test"
            new_name = f"{prefix}_{src.name}"
            shutil.copy(src, dst_dir / new_name)


def main():
    # ── 1. 기존 출력 폴더 초기화 ──────────────────────────
    if OUTPUT_DIR.exists():
        print(f"[초기화] 기존 폴더 삭제: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # ── 2. 이미지 경로 수집 ───────────────────────────────
    random.seed(RANDOM_SEED)

    print("[수집] smoke(input) 이미지 경로 수집 중...")
    smoke_paths    = collect_image_paths("input")

    print("[수집] no_smoke(target) 이미지 경로 수집 중...")
    no_smoke_paths = collect_image_paths("target")

    print(f"  smoke 전체   : {len(smoke_paths)}장")
    print(f"  no_smoke 전체: {len(no_smoke_paths)}장")

    # ── 3. 분할 및 복사 ───────────────────────────────────
    print("\n[복사] smoke → train/smoke & val/smoke")
    split_and_copy(
        smoke_paths,
        train_dst = OUTPUT_DIR / "train" / "smoke",
        val_dst   = OUTPUT_DIR / "val"   / "smoke",
    )

    print("[복사] no_smoke → train/no_smoke & val/no_smoke")
    split_and_copy(
        no_smoke_paths,
        train_dst = OUTPUT_DIR / "train" / "no_smoke",
        val_dst   = OUTPUT_DIR / "val"   / "no_smoke",
    )

    # ── 4. 결과 요약 출력 ────────────────────────────────
    print("\n[완료] 데이터셋 재구성 결과")
    print("=" * 45)
    total = 0
    for split in ("train", "val"):
        for cls in ("smoke", "no_smoke"):
            folder = OUTPUT_DIR / split / cls
            count  = len(list(folder.glob("*")))
            total += count
            print(f"  {split:5s}/{cls:10s} : {count:5d}장")
    print(f"  {'합계':16s} : {total:5d}장")
    print("=" * 45)

    # ── 5. 클래스 불균형 경고 ────────────────────────────
    trn_smoke    = len(list((OUTPUT_DIR / "train" / "smoke").glob("*")))
    trn_no_smoke = len(list((OUTPUT_DIR / "train" / "no_smoke").glob("*")))
    ratio = max(trn_smoke, trn_no_smoke) / max(min(trn_smoke, trn_no_smoke), 1)
    if ratio > 1.5:
        print(f"\n[경고] 클래스 불균형 감지 (비율 {ratio:.2f}x) → pos_weight 조정 고려")
    else:
        print(f"\n[정상] 클래스 균형 양호 (비율 {ratio:.2f}x)")


if __name__ == "__main__":
    main()
