"""
prepare_dataset.py
==================
DesmokeData_dataset → YOLO classification 형식으로 재구성

원본 구조 (단일 플랫 폴더):
  assets/DesmokeData_dataset/
    1.png        ← smoke 이미지 (gt 없음)
    1_gt.png     ← no_smoke 이미지 (gt 붙음)
    2.png
    2_gt.png
    ...

출력 구조:
  data/smoke_cls/
    train/
      smoke/       ← N.png 파일의 80%
      no_smoke/    ← N_gt.png 파일의 80%
    val/
      smoke/       ← N.png 파일의 20%
      no_smoke/    ← N_gt.png 파일의 20%

규칙:
  - 파일명에 '_gt' 포함 → no_smoke
  - 파일명에 '_gt' 미포함 → smoke
  - 원본은 절대 이동하지 않고 shutil.copy 만 사용
  - 실행 전 data/smoke_cls/ 존재 시 삭제 후 재생성
"""

import random
import shutil
from pathlib import Path


# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent          # 프로젝트 루트
SRC_DIR    = BASE_DIR / "assets" / "DesmokeData_dataset"     # 원본 데이터
OUTPUT_DIR = BASE_DIR / "data" / "smoke_cls"                 # 출력 경로

# 재현성 시드 & 분할 비율
RANDOM_SEED = 42
VAL_RATIO   = 0.2


def split_and_copy(src_paths: list[Path], train_dst: Path, val_dst: Path) -> tuple[int, int]:
    """
    src_paths를 8:2로 섞어 분할한 뒤 각 목적지 폴더로 복사한다.
    반환: (train 수, val 수)
    """
    random.shuffle(src_paths)
    n_val   = int(len(src_paths) * VAL_RATIO)
    val_set = src_paths[:n_val]
    trn_set = src_paths[n_val:]

    for dst_dir, file_list in [(train_dst, trn_set), (val_dst, val_set)]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in file_list:
            shutil.copy(src, dst_dir / src.name)

    return len(trn_set), len(val_set)


def main():
    # ── 0. 원본 폴더 존재 확인 ────────────────────────────
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"원본 데이터셋 폴더를 찾을 수 없음: {SRC_DIR}")

    # ── 1. 기존 출력 폴더 초기화 ──────────────────────────
    if OUTPUT_DIR.exists():
        print(f"[초기화] 기존 폴더 삭제: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # ── 2. smoke / no_smoke 분리 ──────────────────────────
    # '_gt'가 파일명에 포함되면 no_smoke, 없으면 smoke
    all_files = list(SRC_DIR.glob("*.png")) + list(SRC_DIR.glob("*.jpg"))

    smoke_paths    = [f for f in all_files if "_gt" not in f.stem]
    no_smoke_paths = [f for f in all_files if "_gt" in f.stem]

    print(f"[수집] smoke(gt 없음) : {len(smoke_paths)}장")
    print(f"[수집] no_smoke(gt)   : {len(no_smoke_paths)}장")

    if not smoke_paths or not no_smoke_paths:
        raise ValueError("smoke 또는 no_smoke 이미지가 0장입니다. 경로/파일명을 확인하세요.")

    # ── 3. 분할 및 복사 ───────────────────────────────────
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

    # ── 4. 결과 요약 출력 ────────────────────────────────
    print("\n[완료] 데이터셋 재구성 결과")
    print("=" * 45)
    print(f"  train/smoke      : {trn_s:5d}장")
    print(f"  train/no_smoke   : {trn_n:5d}장")
    print(f"  val  /smoke      : {val_s:5d}장")
    print(f"  val  /no_smoke   : {val_n:5d}장")
    print(f"  합계             : {trn_s + trn_n + val_s + val_n:5d}장")
    print("=" * 45)

    # ── 5. 클래스 불균형 경고 ────────────────────────────
    ratio = max(trn_s, trn_n) / max(min(trn_s, trn_n), 1)
    if ratio > 1.5:
        print(f"\n[경고] 클래스 불균형 감지 (비율 {ratio:.2f}x) → pos_weight 조정 고려")
    else:
        print(f"\n[정상] 클래스 균형 양호 (비율 {ratio:.2f}x)")

    print(f"\n출력 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
