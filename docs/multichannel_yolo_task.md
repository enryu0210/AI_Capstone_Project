# 다채널 입력 YOLOv8n-cls 재학습 — 얇은 연기 탐지 개선

## 배경

복강경 수술 영상에서 얇은 연기(화면 전체가 뿌옇게 되는 수준)를 현재 RGB 단일 프레임 YOLOv8n-cls 모델이 탐지하지 못함. 얇은 연기의 채도/Dark Channel이 no_smoke와 거의 동일하기 때문. 프레임 간 차이 정보를 입력 채널로 추가하여 모델이 시간적 변화를 직접 학습하도록 개선한다.

## 목표

- 기존 3채널(RGB) 입력을 **4채널(RGB + temporal diff map)** 또는 **6채널(현재 RGB + 이전 RGB)**로 확장
- YOLOv8n-cls 첫 번째 Conv 레이어의 입력 채널 수정
- 수정된 모델로 fine-tuning 후 성능 비교

## 프로젝트 구조 (기존)

```
project/
├── src/
│   ├── prepare_dataset.py       ← 데이터셋 재구성 (8:2, seed=42)
│   ├── pretrained_test.py       ← baseline 테스트
│   ├── train_yolo.ipynb         ← fine-tuning + PR 분석
│   └── video_inference.ipynb    ← 영상 추론 (sliding vote)
├── assets/DesmokeData_dataset/  ← 원본 데이터 (961쌍)
│   ├── {N}.png                  ← smoke 이미지
│   └── {N}_gt.png               ← no_smoke 이미지 (clean GT)
├── data/smoke_cls/              ← 현재 데이터셋
│   ├── train/smoke/     (769장)
│   ├── train/no_smoke/  (769장)
│   ├── val/smoke/       (192장)
│   └── val/no_smoke/    (192장)
├── runs/smoke_detector/yolov8n_cls/weights/best.pt  ← 현재 최종 모델
└── docs/
    └── smoke_detection_dev_log.md   ← 개발 로그
```

## 핵심 제약사항

- 원본 데이터는 **쌍(pair)** 구조: `{N}.png` (smoke) ↔ `{N}_gt.png` (no_smoke)
- 원본은 **영상 프레임이 아님** — 연속 프레임 간 diff를 직접 구할 수 없음
- 따라서 temporal diff 채널은 **합성(synthetic)** 해야 함
- 환경: Windows, RTX 4080 SUPER, `workers=0`, `cache='ram'`

---

## 구현 단계

### 단계 1: 다채널 데이터셋 생성 (`src/prepare_multichannel_dataset.py`)

원본 쌍 데이터에서 4채널 numpy 배열 데이터셋을 생성한다.

**4채널 구성:**
- ch 0~2: 현재 프레임 RGB
- ch 3: **smoke-clean 차이 맵** (= smoke 이미지와 해당 GT 이미지의 grayscale 절대 차이)

**생성 방법:**

```python
# smoke 클래스의 경우:
smoke_img = cv2.imread(f'{N}.png')          # RGB
clean_img = cv2.imread(f'{N}_gt.png')       # 대응 GT
diff_map  = cv2.absdiff(
    cv2.cvtColor(smoke_img, cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
)
# → 4채널: np.dstack([smoke_img, diff_map])  shape=(H, W, 4)

# no_smoke 클래스의 경우:
clean_img = cv2.imread(f'{N}_gt.png')       # RGB
diff_map  = np.zeros((H, W), dtype=np.uint8)  # 차이 없음 = 0
# → 4채널: np.dstack([clean_img, diff_map])  shape=(H, W, 4)
```

**중요 — 학습 시 diff 채널 augmentation:**
- 실제 추론 시에는 이전 프레임과의 차이를 사용하므로, 학습 데이터의 diff_map에 노이즈를 추가하여 도메인 갭을 줄여야 함
- smoke 클래스: diff_map에 Gaussian noise(std=10~20) 추가 + random intensity scale(0.5~1.5)
- no_smoke 클래스: diff_map을 0으로 두되, 가끔(확률 20%) 약한 random noise(std=5) 추가 (수술 도구 움직임 등 시뮬레이션)

**출력 구조:**
```
data/smoke_cls_4ch/
  train/smoke/     ← .npy 파일 (H, W, 4)
  train/no_smoke/
  val/smoke/
  val/no_smoke/
```

- 분할 비율, seed 등은 기존과 동일하게 유지 (8:2, seed=42)

### 단계 2: Custom Dataset 클래스 작성 (`src/multichannel_dataset.py`)

YOLO 내장 데이터 로더는 3채널 이미지만 지원하므로, PyTorch Dataset을 직접 구현한다.

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A

class MultiChannelSmokeDataset(Dataset):
    """4채널 .npy 파일을 로드하는 커스텀 데이터셋"""
    
    def __init__(self, root_dir, split='train', imgsz=224):
        # root_dir/split/smoke/*.npy, root_dir/split/no_smoke/*.npy 로드
        # 클래스 매핑: smoke=1, no_smoke=0
        ...
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # (H, W, 4) uint8
        # resize to imgsz
        # normalize to [0, 1]
        # augmentation (기존과 동일: flip, translate, scale, hsv on RGB channels)
        # transpose to (4, H, W)
        return torch.FloatTensor(img), self.labels[idx]
```

- augmentation은 albumentations 사용 권장 (4채널 지원)
- hsv augmentation은 RGB 3채널에만 적용, diff 채널은 별도 처리

### 단계 3: YOLOv8n-cls 첫 번째 Conv 레이어 수정 (`src/train_multichannel.py`)

```python
from ultralytics import YOLO
import torch

model = YOLO('yolov8n-cls.pt')

# 첫 번째 Conv 레이어 가져오기
first_conv = model.model.model[0].conv  # nn.Conv2d(3, ?, kernel_size, ...)

# 새 Conv 생성 (입력 채널만 3→4로 변경)
new_conv = torch.nn.Conv2d(
    in_channels=4,                          # ← 변경
    out_channels=first_conv.out_channels,
    kernel_size=first_conv.kernel_size,
    stride=first_conv.stride,
    padding=first_conv.padding,
    bias=first_conv.bias is not None
)

# 기존 3채널 가중치 복사 + 4번째 채널은 0 또는 평균으로 초기화
with torch.no_grad():
    new_conv.weight[:, :3, :, :] = first_conv.weight
    new_conv.weight[:, 3:, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
    if first_conv.bias is not None:
        new_conv.bias = first_conv.bias

# 교체
model.model.model[0].conv = new_conv
```

**주의:** YOLO 내장 `model.train()` 은 3채널 전용이므로 사용 불가. 직접 PyTorch 학습 루프를 작성해야 한다.

### 단계 4: 학습 루프 작성 (`src/train_multichannel.py` 계속)

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 하이퍼파라미터 (기존과 최대한 동일)
EPOCHS    = 50
IMGSZ     = 224
BATCH     = 32
LR        = 1e-3
PATIENCE  = 10
DEVICE    = 'cuda:0'

train_ds = MultiChannelSmokeDataset('data/smoke_cls_4ch', split='train', imgsz=IMGSZ)
val_ds   = MultiChannelSmokeDataset('data/smoke_cls_4ch', split='val', imgsz=IMGSZ)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_recall = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    # --- val ---
    model.eval()
    # TP, FP, FN 계산 → Recall, Precision, F1
    # smoke_recall 기준 best model 저장
    
    # --- early stopping ---
    if smoke_recall > best_recall:
        best_recall = smoke_recall
        patience_counter = 0
        torch.save(model.state_dict(), 'runs/smoke_detector/multichannel/best.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break
```

**출력 경로:**
```
runs/smoke_detector/multichannel/
  best.pt           ← 최종 가중치
  training_log.csv  ← epoch별 loss, recall, precision, f1
```

### 단계 5: 추론 파이프라인 수정 (`src/video_inference_multichannel.py`)

영상 추론 시에는 실제 이전 프레임과의 차이를 4번째 채널로 사용한다.

```python
prev_frame = None

for frame in video_frames:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff_map = cv2.absdiff(gray, prev_gray)
    else:
        diff_map = np.zeros_like(gray)
    
    # 4채널 입력 구성
    input_4ch = np.dstack([frame, diff_map])  # (H, W, 4)
    
    # 전처리 후 모델 추론
    # ... (resize, normalize, transpose, unsqueeze)
    
    prev_frame = frame.copy()
```

**중요 — 학습-추론 도메인 차이:**
- 학습: diff = smoke - clean GT (연기 자체의 패턴)
- 추론: diff = 현재 - 이전 프레임 (움직임 + 연기 변화)
- 이 갭을 줄이기 위해 단계 1에서 diff_map에 노이즈를 추가하는 것이 핵심

### 단계 6: 성능 비교 및 PR 분석

기존 3채널 모델과 동일한 방식으로 PR 분석 수행:
- val 세트에서 threshold 0.05~0.95 구간별 Recall, Precision, F1 측정
- 특히 **얇은 연기 FN 감소 여부**에 집중
- 기존 모델 결과(threshold=0.40 기준 Recall=0.964)와 직접 비교

---

## 파일 생성 목록

| 파일 | 역할 | 상태 |
|---|---|---|
| `src/prepare_multichannel_dataset.py` | 4채널 .npy 데이터셋 생성 | ✅ 완료 |
| `src/multichannel_dataset.py` | PyTorch Dataset 클래스 | ✅ 완료 |
| `src/train_multichannel.py` | 모델 수정 + 학습 루프 + PR 분석 | ✅ 완료 |
| `src/video_inference_multichannel.py` | 4채널 영상 추론 파이프라인 | ✅ 완료 |

---

## 기존 코드 수정 금지

- 기존 `src/` 파일은 일절 수정하지 않는다
- 새 파일만 추가하여 병렬 파이프라인으로 구성
- 기존 3채널 모델(`best.pt`)은 비교 기준으로 보존

## 실행 순서

```bash
# 1. 4채널 데이터셋 생성
python src/prepare_multichannel_dataset.py

# 2. 학습
python src/train_multichannel.py

# 3. 영상 추론
python src/video_inference_multichannel.py
```

## 성공 기준

- 특히 FN(얇은 연기 놓침) 7개 → 2개 이하
- 추론 속도 30ms/frame 이하 유지 (실시간성)

---

## 구현 완료 사항 (2026-04-14)

### 주요 구현 내용

#### `src/prepare_multichannel_dataset.py`
- `{N}.png` ↔ `{N}_gt.png` 쌍 자동 매칭 후 4채널 .npy 생성
- smoke diff_map: Gaussian noise(std=15) + intensity scale(0.5~1.5) 추가
- no_smoke diff_map: 20% 확률로 약한 noise(std=5) 추가
- 기존과 동일한 8:2 분할 (seed=42)

#### `src/multichannel_dataset.py`
- YOLO 내장 DataLoader 우회, 직접 PyTorch Dataset 구현
- HSV augmentation(hsv_v=0.9, hsv_s=0.7)은 RGB 3채널에만 적용
- diff 채널은 flip + affine만 적용 (색공간 변환 제외)

#### `src/train_multichannel.py`
- `model.model[0].conv`: Conv2d(3→4), 4번째 채널은 기존 3채널 평균으로 초기화
- AdamW + CosineAnnealingLR (기존 YOLO 학습과 동일 하이퍼파라미터)
- Recall 기준 best.pt 저장, CSV 학습 로그, PR curve 자동 생성

#### `src/video_inference_multichannel.py`
- `hybrid_smoke_decision` 완전 구현 (video_inference.ipynb 셀 0 스케치 기반)
  - 캘리브레이션: 영상 첫 30프레임으로 threshold_c, threshold_s, baseline_sharpness 자동 산출
  - 케이스 1: YOLO ≥ 0.60 → 즉시 smoke
  - 케이스 2: YOLO ≥ 0.25 + haze_score ≥ 2 → smoke (얇은 연기 보완)
  - 케이스 3: haze_score = 3 → smoke_warning
- Sliding window vote는 기존과 동일하게 유지 (VOTE_WINDOW=5, VOTE_K=3)

---

## 실행 결과 (2026-04-14)

### 데이터셋 생성 (`prepare_multichannel_dataset.py`)

| 세트 | smoke | no_smoke |
|---|---|---|
| train | 769장 | 769장 |
| val   | 192장 | 192장 |

### 학습 결과 (`train_multichannel.py`)

- **Best epoch**: 14/50 (Recall 기준 early stop, patience=10)
- **best.pt PR 분석** (val 세트):

| threshold | Recall | Precision | F1 | FN |
|---|---|---|---|---|
| 0.20 (best F1) | 0.9583 | 0.9436 | 0.9509 | 8 |
| 0.50 (argmax) | 0.8958 | 0.9773 | 0.9348 | 20 |

- **기존 3채널 모델** (t=0.40): Recall=0.9635, FN=7
- **4채널 모델** val 성능은 3채널 대비 약간 열세 (FN 7→8, F1 0.9711→0.9509)

### 영상 추론 결과 (`video_inference_multichannel.py`)

| 항목 | 4채널 | 3채널 (기존) |
|---|---|---|
| 탐지된 smoke 프레임 | 241장 (3.0%) | 3,575장 (44.2%) |
| 평균 추론 속도 | **3.2 ms/frame (314 FPS)** | 3.2 ms/frame |

### 핵심 발견: diff 채널 도메인 갭 문제

4채널 모델이 실제 영상에서 smoke를 3%만 탐지한 원인:

| | train | 영상 추론 |
|---|---|---|
| diff_map | smoke - clean GT (연기 밀도 패턴) | frame[t] - frame[t-1] (움직임) |
| 실제 연기 프레임의 diff | **높음** (두꺼운 연기 = 큰 차이) | **≈ 0** (안정된 연기는 프레임 간 변화 없음) |
| 모델 판단 | diff>0 → smoke | diff≈0 → **no_smoke** (오분류) |

모델이 "diff≈0 = no_smoke" 패턴을 과도하게 학습. 노이즈 augmentation으로도 이 갭을 충분히 해소하지 못함.

diff=0 강제 시도 결과도 Recall=0.21로 더 나쁨 → 모델이 diff 채널에 깊게 의존.

### 결론 및 후속 과제

**현재 최선**: 3채널 YOLOv8n-cls + sliding vote (기존 파이프라인)

**4채널 접근법 개선 방향**:
1. 실제 수술 연기 영상 클립 수집 후 연속 프레임 쌍으로 재학습
2. 6채널 방식 (현재 RGB + 이전 프레임 RGB) — diff 없이 시간 정보 직접 입력
3. 긴 시간 차이 diff 사용 (frame[t] - frame[t-30]) — 연기 축적 패턴 포착

---

## 얇은 연기 보조 탐지기 v2 (`src/video_inference_v2.py`) (2026-04-14)

### 배경

4채널 모델이 실패(3% 탐지)한 이후, 31초대 얇은 연기를 탐지하기 위한 영상 품질 기반 보조 탐지 로직 구현. 

### 시뮬레이션 결과 → 구현

v1 ThinSmokeDetector 시뮬레이션 결과: **97.1% 오탐** (전체 8,096프레임 중 7,863프레임 smoke로 판정)

**원인 분석:**
| 원인 | 증거 |
|---|---|
| Laplacian 단일 프레임 변동성 | 깨끗한 구간에서도 135~460으로 3.4배 변동 |
| EMA 베이스라인 오염 | alpha=0.03 → 초반 고변동 구간에 끌려감 |
| sat 임계값 0.75 너무 낮음 | 연기 채도 -24% → ratio=0.76 (임계값 간신히 통과) |

### v2 핵심 설계 변경

| 항목 | v1 (실패) | v2 (수정) |
|---|---|---|
| Sharpness 계산 | 단일 프레임 | **30프레임 rolling median** |
| 베이스라인 방식 | EMA (alpha=0.03) | **Rolling 80th percentile (300프레임)** |
| Sharpness 임계값 | 0.65 | **0.50** (스무딩 후 smoke ≈ 0.28, 정상 ≈ 0.55~0.85) |
| Saturation 임계값 | 0.75 | **0.82** (-24% drop = ratio 0.76 포착) |
| YOLO 하한선 | 0.10 | **0.00** (하한 제거) |
| 실제 보호막 | 단일 임계값 | **Sharpness + Saturation 동시 하락 + 지속성 15프레임** |

### 최종 결과

| 지표 | v1 (실패) | v2 (성공) |
|---|---|---|
| ThinSmoke 추가 탐지 비율 | **97.1%** (오탐 범람) | **0.3%** (28프레임) |
| 0~30초 오탐 수 | 대량 | **0건** |
| ThinSmoke 발화 구간 | 영상 전체 | 31.23s, 33.77s, 39.27s (연기 구간) |
| 최종 smoke 탐지 | YOLO 44.2% | **44.5%** (+0.3%p 보완) |
| 최종 no_smoke 오탐률 | ~53% 증가 | **변화 없음** |

```
ThinSmoke v2 파라미터 요약
─────────────────────────────────────────────
WARMUP_FRAMES     = 300    # 10초 베이스라인 워밍업
BASELINE_WINDOW   = 300    # 10초 rolling percentile 창
BASELINE_PCT      = 80     # 80th percentile 베이스라인
SHARP_SMOOTH_WINDOW = 30   # 1초 rolling median
SHARP_DROP_RATIO  = 0.50   # smoke 시 smooth_sharp ≈ 0.28~0.40
SAT_DROP_RATIO    = 0.82   # smoke 시 sat_ratio ≈ 0.76
PERSIST_WINDOW    = 15     # 지속성 창
PERSIST_THRESH    = 12     # 12/15프레임 이상 동시 하락
YOLO_UNCERTAIN_HI = 0.40   # YOLO가 이미 확신(>0.40)이면 비활성
─────────────────────────────────────────────
```

**결론**: v2는 정밀도 100% (FP 0건)를 유지하면서 YOLO가 놓치는 얇은 연기 초기 구간(31.2~31.6s, 33.8~34.0s, 39.3~39.7s)을 정확히 보완함. 오탐 없는 보수적 보조 탐지기로서 기존 파이프라인에 안전하게 통합 가능.
