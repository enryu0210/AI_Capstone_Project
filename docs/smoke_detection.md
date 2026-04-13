# Smoke Detection - YOLOv8 프로젝트

## 프로젝트 개요
복강경 수술 영상에서 연기를 실시간으로 탐지하는 이진 분류 모델.
- 목적: 연기 발생 시에만 DeSmoking 모델(PFAN)을 트리거
- 핵심 지표: **Recall ≥ 0.99** (연기를 놓치는 것이 가장 위험한 오류)
- 모델: YOLOv8n-cls (ImageNet pretrained → fine-tuning)

## 디렉토리 구조
```
project/
├── CLAUDE.md
├── assets/
│   └── dataset/
│       └── DesmokeData/
│           ├── train/
│           │   ├── input/       ← smoky 이미지 (연기 있음)
│           │   └── target/      ← clean 이미지 (연기 없음)
│           └── test/
│               ├── input/
│               └── target/
├── data/
│   └── smoke_cls/               ← 재구성된 YOLO용 데이터셋 (자동 생성)
│       ├── train/
│       │   ├── smoke/
│       │   └── no_smoke/
│       └── val/
│           ├── smoke/
│           └── no_smoke/
├── runs/
│   └── smoke_detector/          ← 학습 결과 저장
└── src/
    ├── prepare_dataset.py
    ├── pretrained_test.py
    └── train_yolo.py
```

## 핵심 커맨드
```bash
# 1단계: 데이터셋 재구성
python src/prepare_dataset.py

# 2단계: pretrained 모델 추론 테스트 (학습 전)
python src/pretrained_test.py

# 3단계: fine-tuning 학습
python src/train_yolo.py

# 패키지 설치
pip install ultralytics torch torchvision
```

## 개발 지침

### 단계 1 — 데이터셋 재구성 (`src/prepare_dataset.py`)
`assets/dataset/DesmokeData/` 를 YOLO classification 형식으로 재구성한다.

- `input/` → `data/smoke_cls/train/smoke/` 및 `data/smoke_cls/val/smoke/`
- `target/` → `data/smoke_cls/train/no_smoke/` 및 `data/smoke_cls/val/no_smoke/`
- train/val 분할 비율: **8:2**
- 원본 파일은 절대 이동하지 말고 **복사(shutil.copy)** 만 사용
- 실행 전 `data/smoke_cls/` 폴더가 이미 존재하면 삭제 후 재생성
- 완료 후 클래스별 이미지 수를 출력하여 불균형 여부 확인

### 단계 2 — pretrained 추론 테스트 (`src/pretrained_test.py`)
fine-tuning 전, ImageNet pretrained 가중치로 수술 영상 샘플을 먼저 테스트한다.

- `yolov8n-cls.pt` 로드 (ultralytics 자동 다운로드)
- `data/smoke_cls/val/` 에서 smoke/no_smoke 각 50장씩 샘플링하여 추론
- 출력 지표: Accuracy, **Recall (smoke 클래스)**, Precision, F1
- conf threshold: `0.3` (낮게 설정 — 연기 놓치는 것 방지)
- 결과를 보고 Recall ≥ 0.99 충족 여부를 판단

### 단계 3 — fine-tuning 학습 (`src/train_yolo.py`)
pretrained 결과가 Recall 0.99 미달일 때만 실행.

```python
# 핵심 학습 파라미터
model = YOLO('yolov8n-cls.pt')
model.train(
    data='data/smoke_cls',
    epochs=50,
    imgsz=224,
    batch=32,
    device=0,
    patience=10,          # early stopping
    project='runs/smoke_detector',
    name='yolov8n_cls'
)
```

- 학습 완료 후 val confusion matrix 출력
- **smoke 클래스 Recall이 0.99 미달이면** conf threshold를 낮추거나
  pos_weight를 높여서 재학습
- 최종 모델은 `runs/smoke_detector/yolov8n_cls/weights/best.pt` 로 저장됨

## 주의사항
- False Negative(연기 있는데 없다고 판단)가 False Positive보다 훨씬 치명적
  → Precision보다 **Recall을 최우선** 지표로 사용
- 속도도 중요: 추론 시 ms/프레임 반드시 측정하여 기록
- MobileNet 팀과 동일한 val 데이터로 비교해야 공정한 평가 가능
