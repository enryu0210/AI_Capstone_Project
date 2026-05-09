# PFAN De-Smoking PC 클라이언트 — 사용 가이드

복강경 수술 영상의 연기를 PFAN+SurgiATM 모델로 실시간 제거하는 PySide6 데스크톱 앱입니다.
입력은 **웹캠 / USB 캡처카드 / 비디오 파일** 셋 다 지원합니다.

---

## 1. 준비물 (Prerequisites)

| 항목 | 권장값 | 비고 |
|---|---|---|
| OS | Windows 10/11 | 검증 환경 |
| Python | 3.10 ~ 3.12 | Anaconda Python 3.12 에서 검증 완료 |
| GPU (선택) | CUDA 12.x 호환 NVIDIA | GPU 없으면 CPU 모드로 동작 (1~3 FPS) |
| 디스크 | ~3 GB | torch + PySide6 + 의존성 포함 |

---

## 2. 최초 설치 (1회만)

### 2-1. 레포지토리 받기

```powershell
git clone https://github.com/enryu0210/AI_Capstone_Project.git
cd AI_Capstone_Project
git checkout feature/client
```

### 2-2. 파이썬 의존성 설치

```powershell
pip install -r requirements.txt
```

설치되는 핵심 패키지:
- **PyTorch 2.6.0** + **torchvision** — 모델 추론
- **PySide6 6.6.3.1 (LTS 핀)** — GUI. 6.11.x 는 Anaconda Python 3.12 환경에서 DLL 로드 실패 사례가 있어 6.6 으로 고정.
- **pytorch-msssim, einops, timm, mmcv-lite, mmengine** — PFAN 모델 의존성
- **opencv-python, numpy, pillow, matplotlib, ultralytics** — 영상 I/O

### 2-3. 체크포인트 표준 경로로 배치

학습된 가중치(`Final 270_net_G.pth`)를 PFAN 코드가 자동으로 찾는 표준 경로로 복사합니다.

```powershell
python scripts/prepare_checkpoint.py
```

성공 출력 예:
```
[prepare_checkpoint] 프로젝트 루트: F:\...\AI_Capstone_Project
[INFO] 복사 중...
  src: ...\Checkpoint\Final 270_net_G.pth
  dst: ...\Checkpoint\PFAN_Final\270_net_G.pth
[DONE] 체크포인트 준비 완료.
```

> 이 스크립트는 **idempotent** 입니다. 이미 복사돼 있으면 스킵하므로 여러 번 실행해도 안전합니다.

---

## 3. 앱 실행

```powershell
python -m app.main
```

처음 실행하면 PFAN 가중치 로드 + Qt 초기화에 5~10초 정도 걸립니다 (정상).

---

## 4. UI 사용법

```
┌──────────────────────────────────────────────────────────────┐
│ Source: [▾]  Device: [▾]   [Start][Stop][Snapshot][● Record] │
├──────────────────────────────┬───────────────────────────────┤
│         원본 (좌)            │       디스모킹 결과 (우)       │
│                              │                               │
├──────────────────────────────┴───────────────────────────────┤
│ idle │ FPS: -- │ Latency: -- ms │ Frames: 0                  │
└──────────────────────────────────────────────────────────────┘
```

### 4-1. 입력 소스 선택 (Source)

콤보박스에 자동 탐지된 카메라/캡처카드와 **비디오 파일...** 항목이 보입니다.

- **Camera 0** : 첫 번째 카메라/캡처카드. USB 캡처카드 연결 시 가장 흔히 0 또는 1.
- **Camera N** : N 번째 장치 (여러 개 연결 시).
- **비디오 파일...** : 선택하면 파일 다이얼로그가 떠서 mp4/avi/mov/mkv 중 하나 고를 수 있음.

> 마지막에 사용한 카메라/파일은 자동으로 기억됩니다 (다음 실행 시 동일하게 미리 선택됨).

### 4-2. 추론 디바이스 선택 (Device)

| 옵션 | 추론 속도 (256×256 기준) | 권장 상황 |
|---|---|---|
| **GPU (cuda:0)** | 약 60~80 ms/frame (~12~17 FPS) | NVIDIA GPU 가 있는 PC |
| **CPU** | 약 0.5~1 초/frame (~1~2 FPS) | GPU 가 없거나 디버깅용 |

### 4-3. 동작 버튼

| 버튼 | 동작 |
|---|---|
| **Start** | 워커 스레드 시작 → 좌(원본)/우(디스모킹) 듀얼 프리뷰 시작 |
| **Stop** | 워커 정상 종료 (녹화 중이면 mp4 마무리 저장) |
| **Snapshot** | 우측 디스모킹 프리뷰의 **마지막 프레임**을 PNG/JPG 로 저장 |
| **● Record** (toggle) | 다음 Start 클릭 시 mp4 로 녹화 시작. 누른 채로 Start → 저장 위치 묻는 다이얼로그 → 추론 시작과 동시에 녹화 |

### 4-4. 상태바 표시

- **상태** : `idle` / `running` / `running + recording` / `stopping...` / `finished (...)`
- **FPS** : 1초 윈도우 평균 처리 속도
- **Latency** : 마지막 프레임의 추론 소요 시간 (ms)
- **Frames** : 누적 처리 프레임 수

---

## 5. 사용 시나리오 예시

### 시나리오 A — 수술 영상 파일을 디스모킹 후 mp4 로 저장

1. `python -m app.main`
2. Source: **비디오 파일...** → `assets/capture1.avi` 선택
3. Device: **GPU (cuda:0)**
4. **● Record** 토글 ON
5. **Start** 클릭 → 저장 위치 다이얼로그에서 `desmoke_capture1.mp4` 입력
6. 영상이 끝나면 자동으로 "재생이 끝났습니다" 다이얼로그 표시 + mp4 저장 완료

### 시나리오 B — USB 캡처카드를 통해 실시간 디스모킹 미리보기

1. 캡처카드 USB 연결 후 `python -m app.main`
2. Source: **Camera 0**
3. Device: **GPU (cuda:0)**
4. **Start** 클릭 → 좌/우 듀얼 프리뷰 가동
5. 필요한 시점에 **Snapshot** 으로 정지 화면 저장
6. 종료 시 **Stop**

### 시나리오 C — GPU 가 없는 노트북에서 한 장만 비교

1. **Device: CPU** 로 변경
2. Source: **비디오 파일...** → 파일 선택
3. **Start** → FPS 가 1~2 정도 나옴. 첫 몇 프레임만 보고 **Stop**
4. **Snapshot** 으로 디스모킹된 한 장 저장

---

## 6. 자주 만나는 문제 (Troubleshooting)

### Q1. `체크포인트가 준비돼 있지 않습니다` 에러
**원인**: `prepare_checkpoint.py` 를 실행하지 않았거나 원본 `Final 270_net_G.pth` 가 없음.

**해결**: `python scripts/prepare_checkpoint.py` 실행. 그래도 안 되면 다음 경로에 원본이 있는지 확인:
```
src/PFAN-SurgiATM-PGSA/Checkpoint/Final 270_net_G.pth
```

### Q2. `ImportError: DLL load failed while importing QtCore`
**원인**: PySide6 버전이 Python/OS 와 호환되지 않음.

**해결**:
```powershell
pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6
pip install PySide6==6.6.3.1
```

### Q3. `RuntimeError: Error(s) in loading state_dict ... size mismatch`
**원인**: PFAN 모델의 첫 conv 입력 채널이 4 가 아닌 다른 값으로 설정됨. 학습된 가중치는 4채널.

**해결**: `app/desmoke_engine.py` 의 `_TRAIN_OPT_DEFAULTS` 에서 `input_nc=4` 가 그대로인지 확인.

### Q4. 카메라가 콤보박스에 안 뜸
**원인**: OpenCV 가 카메라를 못 잡음 (드라이버 문제, 다른 앱이 점유 등).

**해결**:
- 다른 영상 앱(Zoom, Skype, OBS 등)을 끄고 다시 시도
- USB 캡처카드면 케이블·포트 교체
- Windows 의 카메라 권한 설정 확인 (개인 정보 보호 → 카메라)

### Q5. 출력 영상이 입력과 거의 똑같아 보임
원인: 입력 프레임에 연기가 거의 없음. 모델은 연기가 있는 프레임에서만 가시적인 차이를 만들어내며, 깨끗한 프레임은 거의 그대로 통과시킵니다 (정상 동작).

### Q6. GPU 가 있는데 CPU 모드로 자동 떨어짐
**원인**: PyTorch 가 CUDA 빌드가 아니라 CPU-only 빌드일 가능성.

**해결**:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
`False` 가 나오면 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/) 에서 CUDA 빌드 설치 명령을 따라 재설치.

### Q7. 녹화 mp4 가 재생 안 됨
**원인**: 시스템에 `mp4v` 코덱이 없음.

**해결**: 일반적으로 OpenCV 가 자체 번들이라 거의 문제 없지만, 만약 발생하면 `app/inference_worker.py` 에서 `fourcc` 를 `XVID` 로 바꾸고 확장자를 `.avi` 로 변경.

---

## 7. 파일 구조 한눈에 보기

```
AI_Capstone_Project/
├── app/                              ← PC 클라이언트 본체
│   ├── main.py                       ← 진입점 (python -m app.main)
│   ├── main_window.py                ← PySide6 메인 윈도우
│   ├── desmoke_engine.py             ← PFAN+SurgiATM 추론 래퍼
│   ├── inference_worker.py           ← QThread 워커
│   ├── frame_source.py               ← 카메라/파일 소스 추상화
│   └── settings.py                   ← QSettings 영속화
│
├── scripts/
│   └── prepare_checkpoint.py         ← 체크포인트 표준 경로 복사 (1회)
│
├── src/PFAN-SurgiATM-PGSA/           ← 모델 코드베이스
│   ├── models/                       ← PFAN, SurgiATM, networks 등
│   ├── options/                      ← argparse 정의 (학습/테스트 공용)
│   ├── util/                         ← tensor2im 등 유틸
│   ├── data/                         ← Dataset 정의 (학습 시 사용)
│   ├── Checkpoint/
│   │   ├── Final 270_net_G.pth       ← 학습 완료 가중치 (원본)
│   │   └── PFAN_Final/270_net_G.pth  ← prepare_checkpoint.py 가 만든 표준경로 사본
│   ├── test.py / train.py            ← 원본 학습/테스트 CLI
│   └── README.md
│
├── docs/
│   └── desmoke_client_user_guide.md  ← 이 문서
│
├── requirements.txt                   ← 의존성 핀
└── .gitignore                         ← wandb/, PFAN_InVivo/ 등 제외
```

---

## 8. 개발자용 메모

### 8-1. 추론 파이프라인의 핵심 (`PFAN.forward` 가 끝이 아니다)

PFAN 의 출력 `rho_DNN` 은 **물리 파라미터 맵**이고, 진짜 디스모킹 결과는 SurgiATM 물리 모델을 한 번 더 통과시켜야 나옵니다.

```text
smoky_image → SurgiATM.get_dc → guided_filter → D_refined
            ↓ (cat with smoky)
        PFAN(img, D_refined) → rho_DNN
            ↓
SurgiATM(smoky_image, rho_DNN, D_refined) → pre_clean_image
            ↓ clamp[0,1] → *2−1
        fake_B (최종 출력)
```

`DesmokeEngine.process_tensor` 는 이 전체 흐름을 `Pix2PixModel.forward` 를 통해 자동으로 수행합니다.

### 8-2. 체크포인트 채널 미스매치 주의

학습 시 저장된 `train_opt.txt` 에는 `input_nc: 3` 으로 기록돼 있지만, 실제 가중치
`model_1.0.weight.shape == (64, 4, 1, 1)` 입니다 (PFAN 이 RGB+DCP 를 cat 한 후 conv 통과시키기 때문).

따라서 `app/desmoke_engine.py` 의 `_TRAIN_OPT_DEFAULTS["input_nc"]` 는 반드시 **4** 여야 합니다.

### 8-3. 256×256 고정 추론 이유

모델이 256×256 분포로 학습됐고 BatchNorm 통계가 그 분포에 맞춰져 있어, 다른 해상도로 추론하면 색감/품질이 변합니다. 따라서 추론은 256 으로 고정하고, GUI 표시용으로만 `cv2.resize` 로 원본 해상도 업스케일합니다.

### 8-4. 워커 종료 흐름

```text
사용자 Stop 클릭
    ↓
MainWindow._on_stop() → worker.request_stop() (협력적 종료 플래그)
    ↓
InferenceWorker.run() 의 while 루프 다음 iter 에서 break
    ↓
finally: VideoWriter.release(), FrameSource.release()
    ↓
finished_with_reason("stopped") emit
    ↓
MainWindow._on_worker_finished("stopped") → UI 복귀
```

영상 파일 끝에 도달하면 `request_stop` 없이도 `read()` 가 None 을 반환해 같은 정리 흐름을 탑니다 (`reason="end_of_stream"`).

---

## 9. 다음 단계 아이디어 (Optional)

- **ONNX 변환**: PFAN+SurgiATM 합성을 단일 ONNX 그래프로 export → 의존성 축소
- **고해상도 추론**: 512×512 로 fine-tune 하면 후 GUI 라디오 버튼만 추가
- **SSIM/PSNR 측정 모드**: 정답 영상이 있을 때 실시간 메트릭 표시
- **PIP 모드**: 멀티카메라 수술실 대응
- **smoke 검출 prefilter**: 연기 없는 프레임은 추론 건너뛰기 (속도↑)

---

## 10. 라이선스 / 참고

- 모델: PFAN-SurgiATM-PGSA (자세한 내용은 `src/PFAN-SurgiATM-PGSA/README.md` 와 동봉된 논문 PDF 참고)
- 클라이언트 코드: 캡스톤 프로젝트 산출물
