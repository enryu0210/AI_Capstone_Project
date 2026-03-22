# SmoRestor Notebook 동작 원리 분석 (AI 전공자 수준)

본 문서는 수술용 내시경 이미지의 연기를 제거(Desmoking)하기 위해 작성된 `smorestor_notebook.ipynb`의 동작 과정을 딥러닝 아키텍처 및 학습 파이프라인 관점에서 상세하게 분석합니다.

## 1. 개요 및 문제 정의 (Problem Formulation)
복강경 및 내시경 수술 중 전기소도구 등에 의해 발생하는 연기(Smoke)는 시야를 방해하여 수술의 안정성을 저해합니다. 본 프로젝트는 연기가 포함된 이미지(Smoky)를 입력받아 연기가 없는 깨끗한 이미지(Clean/GT)로 복원하는 Image-to-Image Translation 모델을 학습합니다.
실시간 수술 환경에 적용하기 위해 추론 속도(FPS)와 화질 복원 성능(PSNR, SSIM 등) 간의 균형을 맞추는 하이브리드 아키텍처를 채택하고 있습니다.

## 2. 데이터 파이프라인 (Data Pipeline)
- **FlatPairDataset**: 원본 연기 이미지(`{id}.png`)와 정답 이미지(`{id}_gt.png`) 쌍을 매칭하여 `PyTorch Dataset` 형태로 로드합니다.
- **전처리 (Preprocessing)**: GPU 메모리 효율과 연산 속도 향상을 위해 영상을 특정 해상도(`256x512`)로 Resize하고, 텐서화 및 `[0, 1]` 범위로 정규화(Normalization)합니다.
- **데이터 증강 (Data Augmentation)**: Random Horizontal / Vertical Flip을 확률적으로 적용하여 공간적 대칭성에 강건한 특징을 학습하게 하고 과적합(Overfitting)을 방지합니다.

## 3. 모델 아키텍처 (Model Architecture: SmoRestorLike)
네트워크는 전통적인 U-Net 구조를 뼈대로 삼으며, Bottleneck 구역에 주파수 영역(Frequency Domain) 분해 기법과 Window Attention 메커니즘을 결합한 구조입니다.

### 3.1. Encoder (특징 추출기)
입력 이미지(3채널)를 점진적으로 다운샘플링하여 깊은 수준의 특징(Deep Feature Map)을 추출합니다.
- **ConvNormAct & ResidualConvBlock**: 단순 Conv-ReLU가 아닌, `Conv2d - GroupNorm - GELU` 형태의 블록과 Skip Connection을 더한 잔차 블록(Residual Block)을 사용하여 학습 안정성을 높이고 Gradient Vanishing 문제를 해결합니다. 
- **GroupNorm**: Batch Size가 작을 수밖에 없는 고해상도 영상 처리 태스크에서 BatchNorm보다 안정적인 정규화 효과를 제공합니다.

### 3.2. Frequency Block (주파수 분해 기반 Transformer Bottleneck)
인코더에서 추출된 피처맵은 여러 개의 `FrequencyBlock`을 통과합니다. 연기(Smoke)는 주로 저주파(Low Frequency) 영역에 걸쳐 뿌옇게 나타나고, 장기 표면의 미세한 질감은 고주파(High Frequency) 영역에 존재한다는 도메인 특성에 착안했습니다.
1. **Fourier Prior (주파수 분해)**: 특징 맵을 2D 고속 푸리에 변환(FFT2)하여 주파수 도메인으로 변환합니다. 사전에 정의된 Cutoff Ratio(0.15)를 기준으로 저주파와 고주파 성분을 마스킹(Masking)하여 분리한 후 다시 역변환(iFFT2)합니다. 이후 각각 1x1 Conv 투영(Projection)을 거쳐 다시 결합함으로써, 주파수 대역별 특성을 모델이 스스로 인지하고 필터링하도록 유도합니다.
2. **Window Attention**: ViT(Swin Transformer)와 유사하게 피처 맵을 고정된 크기(Window Size=8)의 패치로 나누어 로컬(Local) Self-Attention을 수행합니다. 이는 기존 Global Attention의 $O(N^2)$ 연산 복잡도 문제를 회피하면서, 안개 낀 영역 내에서 픽셀 간의 의존성(Context)을 효과적으로 파악하게 해줍니다.

### 3.3. Decoder (특징 복원기)
- `ConvTranspose2d`를 사용하여 해상도를 점진적으로 원래 크기까지 복원(Upsampling)합니다.
- **Skip Connection**: U-Net의 핵심인 Skip Connection을 통해, 인코더에서 추출된 특징($f_1, f_2, f_3$)을 디코더의 각 해상도 단계에 Concat 결합합니다. 이는 고해상도의 엣지 및 디테일 손실을 막아줍니다.
- **Global Residual Learning**: 네트워크의 최종 출력 헤드(Sigmoid)를 통과한 값에서 0.5를 뺀 후 원래의 입력 이미지 `x`에 더해주는 형태(`x + (out - 0.5)`)를 취합니다. 즉, 모델은 정답 이미지를 아예 새로 그리는 것이 아니라, **입력 이미지에서 연기를 제거하기 위한 '잔차(Residual)'** 만을 추정하도록 훈련됩니다. 이는 학습 수렴 속도 향상에 크게 기여합니다.

## 4. 손실 함수 (Loss Function)
최상의 시각적 품질을 도출하기 위해 단순한 픽셀 단위 비교(L1 Loss)를 넘어 4가지 목적 함수를 통합 최적화합니다.
1. **L1 Loss ($L_1 = |x - y|$)**: L2(MSE) 대비 Outlier 픽셀에 덜 민감하여 블러링(Blurring) 부작용이 적고 안정적으로 전체 픽셀 복원율을 높입니다.
2. **SSIM Loss ($1 - \text{SSIM}$)**: 이미지의 휘도, 대비, 구조정보를 비교하는 수치인 SSIM을 목적 함수로 사용하여 텍스처와 형태(Structure)가 손상되지 않도록 강제합니다.
3. **VGG Perceptual Loss**: ImageNet으로 사전 학습된 VGG16 네트워크의 특징맵(Feature Map)을 추출하여, 생성된 이미지와 정답 이미지 사이의 "고차원적 의미(Semantic/Perceptual)" 차이를 줄입니다. 눈으로 보기에 더 자연스러운 복원 결과를 만들어냅니다.
4. **Color Loss**: RGB 채널 각각의 전역 평균값 간의 MSE를 계산하여, 조명이나 연기에 의해 전체적인 톤/색감이 왜곡되는 현상을 억제합니다. 내시경 환경의 붉은 조직 색상 유지에 탁월합니다.

## 5. 학습 루프 및 평가지표 (Training & Metrics)
- **Optimizer**: AdamW (기존 Adam에 Weight Decay를 분리 적용하여 일반화 성능 개선)
- **검증 지표 (Validation Metrics)**:
  - **PSNR (Peak Signal-to-Noise Ratio)**: 수치적 화질 평가 지표 (단위: dB)
  - **SSIM (Structural Similarity)**: 구조적 디테일 보존 평가
  - **FADE (Fog Aware Density Evaluator)**: 안개(연기) 밀도 평가 지표(Proxy 구현). Dark Channel Prior 및 색상 채도(Saturation)를 분석해 연기 제거 수준을 무참조(No-reference)로 점수화.
  - **CIEDE2000**: 예측 결과와 실제 정답 조직의 색상이 인간의 눈에 얼마나 유사하게 인지되는지를 LAB 색 공간에서 측정한 고급 색차 지표.

## 6. 실시간 영상 벤치마크 (Real-time Inference)
수술 중 실시간 적용(Real-time Application) 가능성을 검증하기 위해 동영상 스트림 프레임 단위 추론을 실행합니다.
- `torch.no_grad()` 블록 내에서 그래디언트 연산을 비활성화하여 메모리와 속도를 확보합니다.
- `time.perf_counter()` 및 `torch.cuda.synchronize()`를 활용해 GPU 연산 병목까지 고려한 정밀한 프레임당 추론 시간(ms)을 산출합니다.
- 이를 통해 실시간 FPS(Frames Per Second)를 측정하며, 일반적으로 30 FPS 이상 달성 시 수술 로봇/모니터에 지연 없이 즉시 통합할 수 있음을 의미합니다.