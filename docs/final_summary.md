# 복강경 수술 연기 탐지 — 전체 개발 과정 최종 요약

> 작성일: 2026-04-14  
> 브랜치: `feature/detection`  
> 작성자: enryu0210

---

## 프로젝트 목적

복강경 수술 영상에서 연기 발생을 감지하여 DeSmoking 모델(PFAN)을 자동으로 트리거한다.

| 항목 | 내용 |
|---|---|
| 입력 | 복강경 수술 영상 (640×480, 30fps) |
| 출력 | 프레임별 smoke / no_smoke 판정 |
| 핵심 지표 | **Recall 최우선** — 연기 누락이 가장 치명적 |
| 환경 | Windows, RTX 4080 SUPER, Python 3.11, YOLOv8 |

---

## 전체 개발 흐름

```
[데이터셋 준비] → [Baseline 측정] → [Fine-Tuning] → [PR 분석]
       ↓
[영상 추론 v1] → [얇은 연기 문제 발견] → [4채널 실험 (실패)]
       ↓
[얇은 연기 시뮬레이션] → [v1 97% 오탐] → [ThinSmoke v2 구현 (성공)]
```

---

## 단계 1 — 데이터셋 재구성 (`src/prepare_dataset.py`)

### 원본 데이터

```
assets/DesmokeData_dataset/
  {N}.png      ← smoke 이미지 (연기 있는 수술 장면)
  {N}_gt.png   ← no_smoke 이미지 (연기 제거된 clean GT)
  총 961쌍
```

### 분류 기준 및 분할

| 클래스 | 기준 | train | val |
|---|---|---|---|
| smoke | `_gt` 미포함 파일 | 769장 | 192장 |
| no_smoke | `_gt` 포함 파일 | 769장 | 192장 |

- 분할 비율: **8:2**, seed=42 (재현 가능)
- 클래스 균형: 1:1 (완벽 균형)
- `data/`, `assets/` → `.gitignore`에서 제외

---

## 단계 2 — Pretrained Baseline 테스트 (`src/pretrained_test.py`)

Fine-tuning 전 ImageNet pretrained YOLOv8n-cls의 수술 연기 탐지 능력 측정.

| 지표 | 값 |
|---|---|
| Accuracy | 0.51 |
| Recall (smoke) | **0.58** |
| 추론 속도 | 17.0 ms/frame |

→ Recall 0.58, 목표(0.99) 대비 크게 미달 → **fine-tuning 필수** 판정

---

## 단계 3 — Fine-Tuning (`src/train_yolo.ipynb`)

### 최종 학습 파라미터

```python
model.train(
    data    = 'data/smoke_cls',
    epochs  = 50,
    imgsz   = 224,
    batch   = 32,
    device  = 0,
    patience= 10,       # early stopping
    workers = 0,        # Windows spawn 오버헤드 제거
    cache   = 'ram',    # RAM 캐시 → CPU 부하 감소
    hsv_v   = 0.9,      # 밝기 10%~190% 무작위 변화 (조명 불변성)
    hsv_s   = 0.7,
    fliplr  = 0.5,
    flipud  = 0.3,
    translate = 0.1,
    scale   = 0.3,
)
```

### 발생한 문제와 해결

#### 문제 1 — CPU 과부하 (GPU 28%, CPU 80%)
- **원인**: YOLO 기본 `workers=8` → Windows에서 DataLoader 프로세스 8개 동시 생성
- **해결**: `workers=0` + `cache='ram'` 조합으로 GPU 85%까지 끌어올림

#### 문제 2 — 도메인 갭 (핀포인트 조명 오탐)
- **현상**: `capture1.avi` 전 프레임이 smoke로 탐지됨
- **원인**: 모델이 "절대 밝기"로 분류 학습
  ```
  video 프레임 평균 밝기  : 74.2  (어두운 핀포인트 조명)
  train/smoke 평균 밝기  : 133.1
  train/no_smoke 평균 밝기: 85.6
  → 어두운 video = smoke로 오인
  ```
- **시도**: CLAHE 적용 → 역효과 (video가 no_smoke보다 smoke 분포에 더 가까워짐)
- **해결**: `hsv_v=0.9` 극단적 augmentation → 밝기로 구분 불가하게 강제, 연기 고유 텍스처 학습 유도

---

## 단계 4 — Precision-Recall 분석 및 Threshold 최적화

val 세트에서 threshold 0.05~0.95 구간 전체 측정.

| Threshold | Precision | Recall | F1 | FN |
|---|---|---|---|---|
| 0.40 | 0.979 | **0.964** | **0.971** | 7 |
| 0.50 | 0.989 | 0.953 | 0.971 | 9 |
| 0.60 | 0.995 | 0.938 | 0.965 | 12 |
| 0.70 | 1.000 | 0.906 | 0.951 | 18 |

→ **threshold=0.40** 선택: 얇은 연기 포함, Recall·F1 최우선

---

## 단계 5 — 영상 추론 + Sliding Window Vote (`src/video_inference.ipynb`)

### 얇은 연기 탐지 한계 발견

단일 프레임 특성 비교:

| 특성 | 두꺼운 연기 (TP) | 얇은 연기 (FN) | no_smoke |
|---|---|---|---|
| Saturation | 127.18 | **178.28** | **173.27** |
| Dark Channel | 65.21 | **22.54** | **26.53** |

→ 얇은 연기의 채도·DCP가 no_smoke와 거의 동일 → **단일 프레임으로 구별 불가**

### 해결: Sliding Window Vote (시간적 정보 활용)

```python
SMOKE_THRESHOLD = 0.40   # 얇은 연기 포함 (민감하게)
VOTE_WINDOW     = 5      # 최근 5프레임 참조
VOTE_K          = 3      # 3개 이상 smoke → 최종 smoke 판정
```

| 케이스 | Raw 판정 | Vote 결과 |
|---|---|---|
| 두꺼운 연기 | 5/5 smoke | smoke ✓ |
| 얇은 연기 | 3~4/5 smoke | smoke ✓ |
| 단발 오탐 | 1~2/5 smoke | no_smoke ✓ |

### 최종 추론 결과

| 항목 | 값 |
|---|---|
| 처리 프레임 | 8,096장 (270.2초) |
| 평균 속도 | **3.2 ms/frame (311 FPS)** |
| 최종 smoke 탐지 | **3,575장 (44.2%)** |

---

## 단계 6 — 4채널 다채널 실험 (`src/train_multichannel.py` 등)

### 시도 목적

얇은 연기(FN)를 잡기 위해 **RGB(3채널) + temporal diff map(1채널) = 4채널** 입력으로 모델 확장.

### 구현 파일 4종

| 파일 | 역할 |
|---|---|
| `src/prepare_multichannel_dataset.py` | 4채널 .npy 데이터셋 생성 |
| `src/multichannel_dataset.py` | PyTorch Custom Dataset |
| `src/train_multichannel.py` | 4채널 모델 학습 루프 |
| `src/video_inference_multichannel.py` | 4채널 영상 추론 |

### 핵심 수정

```python
# YOLOv8n-cls 첫 번째 Conv 레이어 3→4채널로 교체
first_conv = model.model[0].conv
new_conv = nn.Conv2d(4, first_conv.out_channels, ...)
new_conv.weight[:, :3] = first_conv.weight          # 기존 RGB 가중치 복사
new_conv.weight[:, 3:]  = first_conv.weight.mean(1, keepdim=True)  # 4번째 채널 초기화
model.model[0].conv = new_conv
# 최종 분류 헤드도 1000→2 클래스로 교체
classify_head.linear = nn.Linear(in_features, 2)
```

### 발생한 버그와 해결

| 버그 | 원인 | 해결 |
|---|---|---|
| 한글 경로 로드 실패 | `cv2.imread()` 비ASCII 경로 미지원 | `np.fromfile + cv2.imdecode` 우회 |
| `tuple has no attribute 'argmax'` | ultralytics eval 모드가 tuple 반환 | `extract_tensor()` — tuple이면 `[0]` 반환 |
| 전 epoch Recall=0, TP=0 | 분류 헤드가 여전히 1000 클래스 출력 | `classify_head.linear` 교체 추가 |

### val 성능 비교

| 모델 | threshold | Recall | FN | F1 |
|---|---|---|---|---|
| 3채널 (기존) | 0.40 | 0.9635 | 7 | 0.971 |
| 4채널 | 0.20 | 0.9583 | 8 | 0.9509 |

### 영상 추론 결과 — 도메인 갭 문제

| | 학습 diff | 영상 추론 diff |
|---|---|---|
| 출처 | smoke - clean GT (연기 밀도 패턴) | frame[t] - frame[t-1] (프레임 간 움직임) |
| mean | **47.93** | **4.59** (10.4배 차이) |

→ 안정된 연기 = 프레임 간 변화 없음 = diff ≈ 0 → 모델이 no_smoke로 오분류  
→ 4채널 영상 탐지율: **3.0%** (3채널 44.2% 대비 대폭 하락)  
→ **4채널 접근법 포기**, 기존 3채널 파이프라인 유지

---

## 단계 7 — 얇은 연기 시뮬레이션 (ThinSmoke v1)

### 제안된 알고리즘

31초대 영상을 분석하여 얇은 연기의 물리적 특성 확인:

| 특성 | 30.5s (깨끗) | 31.0s (채도 하락 시작) | 31.5s (선명도 하락) |
|---|---|---|---|
| Saturation | 84.2 | **64.0 (-24%)** | 58.3 |
| Sharpness | 312.7 | 289.1 | **141.2 (-55%)** |

→ 연기 발생 시 Saturation(-24%), Sharpness(-55%) 동시 하락

Dynamic Baseline EMA(alpha=0.03) + Smoke Score 로직 제안.

### 시뮬레이션 결과 — 실패

```
YOLO vote smoke  : 3,575 / 8,096 (44.2%)  ← 정상
ThinSmoke smoke  : 7,863 / 8,096 (97.1%)  ← 오탐 범람
```

**원인 3가지:**
1. Sharpness 단일 프레임이 135~460으로 변동 → 깨끗한 프레임도 ratio=0.28
2. EMA 베이스라인이 초반 고변동 구간에 오염
3. 임계값이 너무 낮아 단일 신호만으로도 탐지

---

## 단계 8 — ThinSmokeDetectorV2 구현 (`src/video_inference_v2.ipynb`)

### v1 → v2 수정 사항

| 문제 | v1 | v2 |
|---|---|---|
| Sharpness 변동성 | 단일 프레임 사용 | **30프레임 rolling median** |
| 베이스라인 오염 | EMA(alpha=0.03) | **300프레임 80th percentile** |
| Saturation 임계값 | 0.75 (연기 ratio=0.76 미탐) | **0.82** |
| Sharpness 임계값 | 0.65 | **0.50** (스무딩 후 기준) |
| YOLO 하한선 | 0.10 (persist 최대 11에서 막힘) | **0.00 (제거)** |
| 실제 FP 방어 | 단일 임계값 | **이중 게이트 + 15프레임 지속성** |

### 알고리즘 흐름

```
[매 프레임]
  ① YOLO 추론 → smoke_conf
  ② Sliding Vote(5프레임) → YOLO 최종 판정
  ③ ThinSmoke v2:
       - sharp_smooth = median(최근 30프레임 sharpness)
       - sharp_base   = 80th percentile(최근 300프레임)
       - sat_base     = 80th percentile(최근 300프레임)
       - both_low = (sharp_smooth/sharp_base < 0.50) AND (sat/sat_base < 0.82)
       - persist_buf에 both_low 결과 추가 (15프레임 창)
       - sum(persist_buf) >= 12 AND YOLO conf < 0.40 → thin_smoke=True
  ④ 최종 = YOLO vote OR thin_smoke
```

### 최종 결과

| 지표 | 값 |
|---|---|
| YOLO smoke | 3,575장 (44.2%) |
| ThinSmoke 추가 | **28장 (0.3%)** |
| 최종 smoke | **3,603장 (44.5%)** |
| 0~30초 오탐 | **0건** |
| ThinSmoke 발화 구간 | 31.23s, 33.77s, 39.27s (전부 연기 구간) |
| 평균 속도 | 3.9 ms/frame (254 FPS) |

### 왜 FP가 0인가

| 상황 | Sharpness | Saturation | both_low |
|---|---|---|---|
| 카메라 이동 | ↓ (1~3프레임) | 유지 | **False** — sat 조건 실패 |
| 수술 도구 교체 | 잠깐 변동 | 잠깐 변동 | **False** — 지속성 15프레임 미달 |
| 깨끗한 구간 | ~0.60-0.85 | ~0.92-1.00 | **False** — 두 조건 모두 미달 |
| 얇은 연기 | ~0.20-0.40 | ~0.75-0.80 | **True** + 15프레임 지속 → 탐지 |

---

## 최종 파이프라인 구조

```
src/
├── prepare_dataset.py          ← 데이터셋 준비 (1회 실행)
├── pretrained_test.py          ← baseline 확인 (참고용)
├── train_yolo.ipynb            ← 모델 학습 (1회 실행)
├── video_inference.ipynb       ← 기본 영상 추론
├── video_inference_v2.ipynb    ← [최신] YOLO + ThinSmoke 보조 탐지
│
├── prepare_multichannel_dataset.py  ← 4채널 실험용
├── multichannel_dataset.py          ← 4채널 실험용
├── train_multichannel.py            ← 4채널 실험용 (결과: 열세)
├── video_inference_multichannel.py  ← 4채널 실험용 (결과: 3% 탐지)
└── video_inference_v2.py            ← ipynb 변환 전 원본 스크립트
```

### 실행 순서 (신규 영상 탐지 시)

```bash
# 학습이 이미 완료된 경우 → 3단계만 실행
# 1. (최초 1회) 데이터셋 준비
python src/prepare_dataset.py

# 2. (최초 1회) 모델 학습
# src/train_yolo.ipynb 실행

# 3. 영상 추론 (VIDEO_PATH 수정 후)
# src/video_inference_v2.ipynb 실행
```

---

## 성능 지표 최종 요약

| 버전 | Recall (val) | 영상 smoke% | 오탐 주요 원인 |
|---|---|---|---|
| Pretrained (학습 전) | 0.58 | - | ImageNet 분포 차이 |
| Fine-tuned v1 (threshold=0.60) | 0.938 | - | 얇은 연기 미탐 |
| Fine-tuned + threshold=0.40 | **0.964** | 44.2% | - |
| + Sliding Vote (5/3) | 0.964 | 44.2% | 단발 FP 제거 |
| 4채널 모델 | 0.9583 | 3.0% | diff 도메인 갭 |
| **v2 (YOLO + ThinSmoke)** | **0.964+** | **44.5%** | **FP 0건** |

---

## 핵심 교훈

| 항목 | 내용 |
|---|---|
| **CLAHE 역효과** | 핀포인트 조명 영상에 CLAHE 적용 시 smoke 분포에 더 가까워짐 |
| **DCP 단독 불가** | 핀포인트 조명이 DCP 값을 올려 연기와 구분 불가 |
| **4채널 도메인 갭** | 학습 diff(smoke-GT)와 추론 diff(프레임 간)의 10배 차이 → 근본적 한계 |
| **Sharpness 변동성** | Laplacian 분산은 단일 프레임에서 3배 이상 변동 → rolling median 필수 |
| **지속성 조건** | 연기는 수십 프레임 지속, 카메라 이동은 1~3프레임 → persist 15프레임이 핵심 필터 |
| **이중 게이트** | Sharpness + Saturation 동시 하락 요구 → 카메라 이동(Sharpness만 ↓) 방어 |
| **Windows 학습** | `workers=0` + `cache='ram'` 조합이 가장 효율적 |
