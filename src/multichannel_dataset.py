"""
multichannel_dataset.py
========================
4채널 .npy 파일을 로드하는 PyTorch Dataset 클래스.

YOLO 내장 DataLoader는 3채널 전용이므로,
4채널(RGB + diff_map)을 지원하기 위해 직접 구현.

클래스 매핑:
  no_smoke = 0  (알파벳 순서 → YOLO 학습 시 디렉토리 정렬과 일치)
  smoke    = 1

Augmentation (train split):
  - Horizontal / Vertical Flip
  - Affine (Translate ±10%, Scale 70~130%)
  - HSV 변환: RGB 3채널에만 적용 (hsv_v=0.9, hsv_s=0.7)
  - diff 채널: flip + affine만 적용, 색공간 변환 제외
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union


class MultiChannelSmokeDataset(Dataset):
    """
    data/smoke_cls_4ch/{split}/{smoke,no_smoke}/*.npy 를 로드하는 데이터셋.

    Args:
        root_dir: data/smoke_cls_4ch 경로 (str 또는 Path)
        split   : 'train' 또는 'val'
        imgsz   : 리사이즈 크기 (정방형, 기본 224)
        augment : augmentation 여부 (train=True 권장, val=False)
    """

    # 알파벳 순서 → YOLO 디렉토리 정렬과 동일하게 유지
    CLASS_MAP = {"no_smoke": 0, "smoke": 1}

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        imgsz: int = 224,
        augment: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split    = split
        self.imgsz    = imgsz
        # val은 augmentation 강제 제외 (순수 평가용)
        self.augment  = augment and (split == "train")

        self.files  = []
        self.labels = []

        for cls_name, label in self.CLASS_MAP.items():
            cls_dir = self.root_dir / split / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"데이터셋 폴더 없음: {cls_dir}")
            for npy_path in sorted(cls_dir.glob("*.npy")):
                self.files.append(npy_path)
                self.labels.append(label)

        print(
            f"[Dataset] {split:5s}: smoke={self.labels.count(1)}장, "
            f"no_smoke={self.labels.count(0)}장, augment={self.augment}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tensor: (4, H, W) float32, [0, 1] 정규화됨
            label : int (0=no_smoke, 1=smoke)
        """
        data = np.load(str(self.files[idx]))  # (H, W, 4) uint8

        # ch 0~2: RGB  /  ch 3: diff_map
        rgb  = data[:, :, :3]
        diff = data[:, :, 3]

        # ── 리사이즈 ──────────────────────────────────────────
        rgb  = cv2.resize(rgb,  (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        diff = cv2.resize(diff, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            rgb, diff = self._augment(rgb, diff)

        # ── 정규화 [0,255] → [0,1] ────────────────────────────
        rgb_norm  = rgb.astype(np.float32)  / 255.0
        diff_norm = diff.astype(np.float32) / 255.0

        # (H, W, 4) → (4, H, W)
        img_4ch = np.dstack([rgb_norm, diff_norm[:, :, np.newaxis]])  # (H, W, 4)
        img_4ch = img_4ch.transpose(2, 0, 1)                           # (4, H, W)

        return torch.FloatTensor(img_4ch), self.labels[idx]

    # ──────────────────────────────────────────────────────────
    #  Augmentation 내부 메서드
    # ──────────────────────────────────────────────────────────

    def _augment(
        self,
        rgb: np.ndarray,
        diff: np.ndarray,
    ) -> tuple:
        """
        RGB와 diff에 동일한 기하학적 변환(flip, affine)을 적용하고,
        RGB에만 HSV 색공간 변환을 추가로 적용한다.

        기하학적 변환: RGB와 diff 동일하게 → 공간 정합 유지
        HSV 변환    : RGB만 → diff_map은 강도값이므로 색공간 변환 불필요
        """
        H, W = rgb.shape[:2]

        # ── Horizontal Flip (50%) ──────────────────────────────
        if np.random.rand() < 0.5:
            rgb  = np.fliplr(rgb).copy()
            diff = np.fliplr(diff).copy()

        # ── Vertical Flip (30%) ────────────────────────────────
        if np.random.rand() < 0.3:
            rgb  = np.flipud(rgb).copy()
            diff = np.flipud(diff).copy()

        # ── Affine (Translate + Scale) ─────────────────────────
        # translate: ±10%,  scale: 70%~130%
        tx    = np.random.uniform(-0.1, 0.1) * W
        ty    = np.random.uniform(-0.1, 0.1) * H
        scale = np.random.uniform(0.7, 1.3)
        cx, cy = W / 2, H / 2

        M    = cv2.getRotationMatrix2D((cx, cy), angle=0, scale=scale)
        M[0, 2] += tx
        M[1, 2] += ty

        rgb  = cv2.warpAffine(rgb,  M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        diff = cv2.warpAffine(diff, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

        # ── HSV Augmentation (RGB 채널에만) ────────────────────
        # hsv_v=0.9 (밝기), hsv_s=0.7 (채도) → 조명 불변성 확보
        # 기존 3채널 모델과 동일한 augmentation 강도 유지
        rgb_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue: ±18도 (±0.1 * 180)
        rgb_hsv[:, :, 0] = np.clip(
            rgb_hsv[:, :, 0] + np.random.uniform(-18, 18), 0, 180
        )
        # Saturation: × (1 ± 0.7)
        sat_scale = np.random.uniform(1 - 0.7, 1 + 0.7)
        rgb_hsv[:, :, 1] = np.clip(rgb_hsv[:, :, 1] * sat_scale, 0, 255)
        # Value (밝기): × (1 ± 0.9) → 10%~190% 범위
        val_scale = np.random.uniform(1 - 0.9, 1 + 0.9)
        rgb_hsv[:, :, 2] = np.clip(rgb_hsv[:, :, 2] * val_scale, 0, 255)

        rgb = cv2.cvtColor(rgb_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return rgb, diff
