"""
prepare_multichannel_dataset.py
================================
원본 쌍(pair) 데이터에서 4채널 numpy 배열 데이터셋을 생성한다.

4채널 구성:
  ch 0~2: RGB 이미지
  ch 3  : smoke-clean 차이 맵 (grayscale absdiff)
           - smoke 클래스: absdiff(smoke_gray, clean_gray)  → 연기 고유 패턴
           - no_smoke 클래스: 0으로 채운 맵 (차이 없음)

학습-추론 도메인 갭 줄이기 (augmentation):
  - smoke diff_map: Gaussian noise(std=15) + random intensity scale(0.5~1.5)
    → 추론 시 "이전 프레임 diff"와 도메인이 달라도 학습된 것처럼 작동하도록
  - no_smoke diff_map: 20% 확률로 약한 noise(std=5) 추가
    → 실제 수술 도구/조직 움직임으로 발생하는 미세 diff 시뮬레이션

출력 구조:
  data/smoke_cls_4ch/
    train/smoke/     ← .npy 파일 (H, W, 4) uint8
    train/no_smoke/
    val/smoke/
    val/no_smoke/
"""

import random
import shutil
import numpy as np
import cv2
from pathlib import Path


def imread_unicode(path: Path) -> np.ndarray:
    """
    Windows에서 한글 등 비ASCII 경로를 지원하는 이미지 로더.
    cv2.imread()는 Windows에서 비ASCII 경로를 처리하지 못하므로
    np.fromfile + cv2.imdecode 방식으로 우회한다.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
SRC_DIR    = BASE_DIR / "assets" / "DesmokeData_dataset"
OUTPUT_DIR = BASE_DIR / "data" / "smoke_cls_4ch"

RANDOM_SEED = 42
VAL_RATIO   = 0.2


def add_noise_to_diffmap(
    diff_map: np.ndarray,
    std: float,
    scale_range: tuple = (1.0, 1.0),
) -> np.ndarray:
    """
    diff_map에 Gaussian noise와 intensity scaling을 추가한다.
    추론 시 "이전 프레임 diff" 특성(노이즈 포함)을 학습 데이터에 반영하기 위함.
    """
    noise  = np.random.normal(0, std, diff_map.shape).astype(np.float32)
    scale  = np.random.uniform(*scale_range)
    result = diff_map.astype(np.float32) * scale + noise
    # [0, 255] 범위로 클리핑 후 uint8 변환
    return np.clip(result, 0, 255).astype(np.uint8)


def build_4ch_smoke(
    smoke_path: Path,
    clean_path: Path,
    augment: bool = True,
) -> np.ndarray:
    """
    smoke 이미지 + clean GT → 4채널 배열.
    ch 0~2: smoke RGB  /  ch 3: absdiff (연기 고유 패턴)

    augment=True → diff_map에 노이즈 추가 (학습-추론 도메인 갭 완화)
    """
    smoke_img = imread_unicode(smoke_path)
    clean_img = imread_unicode(clean_path)
    if smoke_img is None or clean_img is None:
        raise FileNotFoundError(
            f"이미지 로드 실패: {smoke_path.name} 또는 {clean_path.name}"
        )

    smoke_gray = cv2.cvtColor(smoke_img, cv2.COLOR_BGR2GRAY)
    clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    diff_map   = cv2.absdiff(smoke_gray, clean_gray)  # 연기 고유 패턴

    if augment:
        # 추론 시 "이전 프레임 diff"와의 도메인 갭을 줄이기 위한 노이즈 추가
        # → 완벽한 smoke-clean diff가 아닌 noisy한 차이도 학습하도록
        diff_map = add_noise_to_diffmap(diff_map, std=15, scale_range=(0.5, 1.5))

    # BGR → RGB 변환 후 4채널로 결합
    smoke_rgb = cv2.cvtColor(smoke_img, cv2.COLOR_BGR2RGB)
    return np.dstack([smoke_rgb, diff_map])  # (H, W, 4) uint8


def build_4ch_no_smoke(
    clean_path: Path,
    augment: bool = True,
) -> np.ndarray:
    """
    no_smoke GT 이미지 → 4채널 배열.
    ch 0~2: clean RGB  /  ch 3: 0 (연기가 없으므로 차이 없음)

    augment=True → 20% 확률로 약한 노이즈 추가
      → 수술 도구/조직 움직임으로 인한 미세한 diff 시뮬레이션
    """
    clean_img = imread_unicode(clean_path)
    if clean_img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {clean_path.name}")

    H, W     = clean_img.shape[:2]
    diff_map = np.zeros((H, W), dtype=np.uint8)

    if augment and np.random.rand() < 0.20:
        diff_map = add_noise_to_diffmap(diff_map, std=5)

    clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    return np.dstack([clean_rgb, diff_map])  # (H, W, 4) uint8


def save_split(pairs: list, split: str, augment: bool) -> int:
    """
    pairs의 각 (smoke_path, clean_path)에서 4채널 .npy를 생성해 저장.
    smoke 1장 + no_smoke 1장 = 2개의 .npy 파일이 생성됨.
    """
    smoke_dir    = OUTPUT_DIR / split / "smoke"
    no_smoke_dir = OUTPUT_DIR / split / "no_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    no_smoke_dir.mkdir(parents=True, exist_ok=True)

    for smoke_path, clean_path in pairs:
        stem = smoke_path.stem  # e.g. "1"

        # smoke 4채널 저장
        arr_smoke = build_4ch_smoke(smoke_path, clean_path, augment=augment)
        np.save(str(smoke_dir / f"{stem}.npy"), arr_smoke)

        # no_smoke 4채널 저장
        arr_no_smoke = build_4ch_no_smoke(clean_path, augment=augment)
        np.save(str(no_smoke_dir / f"{stem}_gt.npy"), arr_no_smoke)

    return len(pairs)


def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"원본 데이터셋 폴더 없음: {SRC_DIR}")

    if OUTPUT_DIR.exists():
        print(f"[초기화] 기존 폴더 삭제: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # ── 쌍 매칭 ({N}.png ↔ {N}_gt.png) ──────────────────────
    all_files   = list(SRC_DIR.glob("*.png"))
    smoke_paths = sorted([f for f in all_files if "_gt" not in f.stem])

    pairs = []
    for sp in smoke_paths:
        cp = SRC_DIR / f"{sp.stem}_gt.png"
        if cp.exists():
            pairs.append((sp, cp))
        else:
            print(f"[경고] GT 없음, 스킵: {sp.name}")

    print(f"[수집] 유효 쌍 수: {len(pairs)}쌍")
    if not pairs:
        raise ValueError("유효한 쌍이 없습니다. SRC_DIR 경로를 확인하세요.")

    # ── 8:2 분할 (seed=42, 기존 3채널 데이터셋과 동일) ──────
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n_val     = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:n_val]
    trn_pairs = pairs[n_val:]

    print(f"\n[생성] train 세트 ({len(trn_pairs)}쌍, augment=True) ...")
    save_split(trn_pairs, split="train", augment=True)

    # val 세트는 augmentation 없이 (순수 평가용)
    print(f"[생성] val 세트 ({len(val_pairs)}쌍, augment=False) ...")
    save_split(val_pairs, split="val", augment=False)

    # ── 결과 요약 ──────────────────────────────────────────
    print("\n[완료] 4채널 데이터셋 생성 결과")
    print("=" * 45)
    for split in ("train", "val"):
        for cls in ("smoke", "no_smoke"):
            count = len(list((OUTPUT_DIR / split / cls).glob("*.npy")))
            print(f"  {split:5s}/{cls:10s} : {count:5d}장")
    print("=" * 45)
    print(f"\n출력 경로: {OUTPUT_DIR}")

    # 첫 번째 .npy 파일로 shape 검증
    sample = np.load(str(next((OUTPUT_DIR / "train" / "smoke").glob("*.npy"))))
    print(f"샘플 shape: {sample.shape}  dtype: {sample.dtype}")
    assert sample.shape[2] == 4, "4채널이어야 합니다."
    print("[검증] shape 정상 (H, W, 4)")


if __name__ == "__main__":
    main()
