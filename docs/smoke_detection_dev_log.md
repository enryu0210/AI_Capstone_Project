# Smoke Detection 개발 로그

YOLOv8n-cls 기반 복강경 수술 영상 연기 탐지 모델 개발 전 과정 기록.

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 목적 | 복강경 수술 영상에서 연기 발생 감지 → DeSmoking 모델(PFAN) 트리거 |
| 모델 | YOLOv8n-cls (ImageNet pretrained → fine-tuning) |
| 핵심 지표 | Recall ≥ 0.99 (연기 누락이 가장 치명적) |
| 브랜치 | `feature/detection` |

---

## 단계 1 — 데이터셋 재구성 (`src/prepare_dataset.py`)

### 원본 데이터 구조

```
assets/DesmokeData_dataset/
  1.png        ← smoke 이미지 (연기 있음)
  1_gt.png     ← no_smoke 이미지 (연기 없는 clean GT)
  2.png / 2_gt.png ...  (총 961쌍)
```

### 분류 규칙

- 파일명에 `_gt` 포함 → **no_smoke**
- 파일명에 `_gt` 미포함 → **smoke**

### 출력 구조 (`data/smoke_cls/`)

```
train/smoke/     769장
train/no_smoke/  769장
val/smoke/       192장
val/no_smoke/    192장
합계             1,922장
```

- 분할 비율: 8:2 (random seed=42)
- 클래스 균형: 1.00x (완벽 균형)
- 원본 파일은 `shutil.copy`로 복사만 (이동 금지)

### 주의사항

- `data/`, `assets/` 는 `.gitignore`에 추가하여 git 업로드 제외
- `*.pt` 가중치 파일도 git 제외

---

## 단계 2 — Pretrained Baseline 테스트 (`src/pretrained_test.py`)

fine-tuning 전, ImageNet pretrained 가중치의 수술 연기 탐지 능력을 측정.

### 결과

| 지표 | 값 |
|---|---|
| Accuracy | 0.51 |
| Precision | 0.51 |
| **Recall (smoke)** | **0.58** |
| F1 | 0.54 |
| 추론 속도 | 17.0 ms/frame |

→ Recall 0.58로 목표(0.99) 크게 미달. **fine-tuning 필요** 판정.

---

## 단계 3 — Fine-Tuning (`src/train_yolo.ipynb`)

### 최종 학습 파라미터

```python
model.train(
    data      = 'data/smoke_cls',
    epochs    = 50,
    imgsz     = 224,
    batch     = 32,
    device    = 0,          # RTX 4080 SUPER
    patience  = 10,         # early stopping
    workers   = 0,          # Windows spawn 오버헤드 제거
    cache     = 'ram',      # 이미지 RAM 캐시 → CPU 부하 감소

    # 조명 불변성 확보 augmentation
    hsv_v     = 0.9,        # 밝기 10%~190% 무작위 변화
    hsv_s     = 0.7,        # 채도 변화
    fliplr    = 0.5,
    flipud    = 0.3,
    translate = 0.1,
    scale     = 0.3,
)
```

### 학습 과정에서 발생한 문제와 해결

#### 문제 1 — CPU 과부하 (GPU 28%, CPU 80%)

- **원인**: YOLO 기본 `workers=8` → Windows에서 DataLoader 프로세스 8개 동시 실행
- **해결**: `workers=0` (단일 프로세스) + `cache='ram'` (RAM 캐시로 I/O 제거)

#### 문제 2 — 도메인 갭 (핀포인트 조명 오탐)

- **현상**: `capture1.avi`의 모든 프레임이 smoke로 탐지됨
- **원인 분석**:
  ```
  video 프레임 밝기  : 74.2  (어두운 핀포인트 조명)
  train/smoke 밝기  : 133.1
  train/no_smoke 밝기: 85.6
  ```
  모델이 "절대 밝기/조명 스타일"로 분류 → 어두운 영상 = smoke로 오인
- **시도 1 — CLAHE 적용**: 역효과 (video mean이 no_smoke에서 더 멀어짐)
- **해결**: `hsv_v=0.9` 극단적 augmentation → 모델이 밝기로 구분 불가하게 강제, 연기 고유 텍스처 학습 유도

---

## 단계 4 — Precision-Recall 분석 및 Threshold 최적화

### 분석 방법

val 세트 전체의 smoke confidence를 수집, threshold 0.05~0.95 구간별 지표 측정.

### 결과 (주요 구간)

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.40 | 0.979 | 0.964 | **0.971** | 4 | 7 |
| 0.50 | 0.989 | 0.953 | 0.971 | 2 | 9 |
| 0.60 | 0.995 | 0.938 | 0.965 | 1 | 12 |
| 0.70 | 1.000 | 0.906 | 0.951 | 0 | 18 |

- PR curve 및 threshold vs metrics 차트: `runs/smoke_detector/yolov8n_cls/pr_curve.png`

---

## 단계 5 — DCP(Dark Channel Prior) 적용 가능성 검토

### DCP 점수 비교

| 데이터 | DCP mean | std |
|---|---|---|
| train/smoke | 74.94 | 44.79 |
| train/no_smoke | 19.63 | 11.80 |
| video frames | 49.93 | 13.66 |

### 결론: DCP 단독 사용 불가

- **no_smoke 탐지**: 정확도 100% (DCP < 20이면 확실히 연기 없음)
- **smoke 탐지**: 정확도 64% — smoke의 분산이 너무 커서 불안정
- **핀포인트 조명**: DCP가 조명의 밝은 spot을 연기로 오인 (video 70th pct=87 > smoke 70th pct=73)
- **결론**: 외부 조명 환경이 DCP에 영향을 주므로 이 데이터셋/환경에서는 단독 적용 불가

---

## 단계 6 — 얇은 연기 탐지 한계 분석 및 Sliding Window Vote

### 얇은 연기 탐지 불가 원인

FN(놓친 얇은 연기)과 no_smoke의 시각적 특성 비교:

| 특성 | 두꺼운 연기 (TP) | 얇은 연기 (FN) | no_smoke |
|---|---|---|---|
| Saturation | 127.18 | **178.28** | **173.27** |
| Dark Channel | 65.21 | **22.54** | **26.53** |
| Sharpness (Lap.var) | 224.4 | 240.8 | 213.4 |

→ **얇은 연기의 채도/Dark Channel이 no_smoke와 거의 동일**  
→ 단일 프레임 이미지 특성(DCP, 채도, 선명도)으로는 구별 불가능한 근본적 한계

### 해결책: Sliding Window Vote (시간적 정보 활용)

연기의 시간적 특성 활용:
- **연기**: 수~수십 프레임 연속 지속
- **단발 오탐(FP)**: 1~2프레임에 그침

```python
SMOKE_THRESHOLD = 0.40   # 얇은 연기 포함 (민감하게)
VOTE_WINDOW     = 5      # 최근 5프레임 참조
VOTE_K          = 3      # 3개 이상 smoke이면 최종 smoke 판정
```

| 케이스 | Raw 판정 | Vote 결과 |
|---|---|---|
| 두꺼운 연기 | 5/5 smoke | 통과 → smoke |
| 얇은 연기 | 3~4/5 smoke | 통과 → smoke |
| 핀포인트 조명 오탐 | 1~2/5 smoke | 탈락 → no_smoke |

---

## 실제 영상 추론 (`src/video_inference.ipynb`)

### 입력 영상 정보

| 항목 | 값 |
|---|---|
| 파일 | `assets/capture1.avi` |
| 해상도 | 640 × 480 |
| FPS | 29.97 |
| 총 프레임 | 8,097장 (약 4분 30초) |

### 노트북 구성

| 셀 | 내용 |
|---|---|
| 셀 0 | 경로·파라미터 설정, 모델 로드 |
| 셀 1 | 전체 프레임 추론 + sliding vote → 오버레이 영상 저장 |
| 셀 2 | 5초 단위 smoke 탐지 비율 타임라인 차트 |
| 셀 3 | smoke/no_smoke 대표 샘플 프레임 시각화 |

### 출력 파일

```
runs/smoke_detector/video_result/
  capture1_result.avi      ← 오버레이 영상
  smoke_timeline.png       ← 시간대별 smoke 비율 차트
  sample_frames.png        ← 대표 샘플 프레임
```

---

## 파일 구조

```
project/
├── src/
│   ├── prepare_dataset.py   ← 단계 1: 데이터셋 재구성
│   ├── pretrained_test.py   ← 단계 2: baseline 테스트
│   ├── train_yolo.ipynb     ← 단계 3~4: fine-tuning + PR 분석
│   └── video_inference.ipynb← 단계 5~6: 영상 추론 (sliding vote)
├── data/smoke_cls/          ← 재구성된 데이터셋 (git 제외)
├── runs/smoke_detector/     ← 학습 결과 (git 제외)
│   └── yolov8n_cls/
│       └── weights/best.pt  ← 최종 모델
└── docs/
    ├── smoke_detection.md           ← 프로젝트 설계 문서
    └── smoke_detection_dev_log.md   ← 개발 로그 (본 문서)
```

---

## 실행 순서

```bash
# 1. 데이터셋 재구성
python src/prepare_dataset.py

# 2. Baseline 테스트 (선택)
python src/pretrained_test.py

# 3. Fine-tuning + PR 분석
# → src/train_yolo.ipynb 셀 순서대로 실행

# 4. 영상 추론
# → src/video_inference.ipynb 셀 순서대로 실행
```

---

## 주요 발견 및 교훈

| 항목 | 내용 |
|---|---|
| 데이터셋 한계 | `_gt` (no_smoke)가 처리된 clean 이미지 → 실제 수술 영상과 분포 차이 존재 |
| CLAHE 역효과 | 핀포인트 조명 영상에 CLAHE 적용 시 no_smoke보다 smoke 분포에 더 가까워짐 |
| DCP 한계 | 핀포인트 조명이 DCP 값을 올려 연기와 구분 불가 |
| 얇은 연기 | 채도/DCP가 no_smoke와 동일 → 단일 프레임 특성으로 구별 불가, 시간적 정보 필요 |
| GPU 활용 | workers=0 + cache=ram 조합이 Windows에서 가장 효율적 |
