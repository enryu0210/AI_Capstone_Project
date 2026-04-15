# video_inference_v3.ipynb 작업 결과

> 작업일: 2026-04-15  
> 기반 문서: `docs/thin_smoke_continuity_fix.md`  
> 수정 대상: `src/video_inference_v3.ipynb` (신규 생성)  
> 추가 수정: BalloonGate (흰 기구 오탐 억제) — 2026-04-15

---

## 개요

v2에서 얇은 연기 구간의 탐지 끊김(flickering) 문제를 해결하기 위해 3단계 연속성 개선을 적용하였다.

**원칙**: 오탐(FP) 허용, 미탐(FN) 불허

---

## 변경 사항 (3단계)

### 1단계 — Cooldown (최소 유지 시간)

smoke 판정이 한 번 발생하면 `COOLDOWN_FRAMES` 동안 smoke 상태를 강제 유지한다.

| 파라미터 | 값 | 의미 |
|---|---|---|
| `COOLDOWN_FRAMES` | `45` | 30fps 기준 약 1.5초 강제 유지 |

**구현 위치**: Hysteresis 판정 이후 (`final_smoke = smoke_state` 다음)

```python
if final_smoke:
    cooldown_counter = COOLDOWN_FRAMES
elif cooldown_counter > 0:
    final_smoke = True
    cooldown_counter -= 1
    cooldown_forced_count += 1
```

---

### 2단계 — ThinSmoke 지속성 조건 완화

| 파라미터 | v2 | v3 | 변경 이유 |
|---|---|---|---|
| `PERSIST_WINDOW` | `15` | `30` | 더 긴 구간 관찰 |
| `PERSIST_THRESH` | `12` (80%) | `15` (50%) | 조건 완화 — flickering 구간 대응 |

---

### 3단계 — Hysteresis (이력 임계값)

진입과 해제에 다른 조건을 적용하여 경계선 flickering 제거.

| 조건 | 기준 |
|---|---|
| **진입** (no_smoke → smoke) | `yolo_vote or thin_smoke` (기존 동일) |
| **해제** (smoke → no_smoke) | 아래 3조건을 `EXIT_CONSECUTIVE`(30) 프레임 연속 만족 시만 해제 |

해제 조건 파라미터:

| 파라미터 | 값 | 의미 |
|---|---|---|
| `EXIT_CONSECUTIVE` | `30` | 30프레임(약 1초) 연속 깨끗해야 해제 |
| `EXIT_YOLO_CONF` | `0.15` | YOLO confidence 상한 |
| `EXIT_SHARP_RATIO` | `0.90` | sharpness ratio 하한 |
| `EXIT_SAT_RATIO` | `0.95` | saturation ratio 하한 |

---

## 추론 파이프라인 순서

```
① YOLO 추론
② Sliding window vote (YOLO)
③ ThinSmoke v2 보조 탐지 (PERSIST 완화 적용)
④ Hysteresis 판정 (진입 쉽게, 해제 어렵게)
⑤ Cooldown (smoke_state 기준 최소 45프레임 강제 유지)
⑥ 오버레이 저장 + CSV 로그
```

> **순서 주의**: Cooldown은 반드시 Hysteresis 결과(`smoke_state`) 기준으로 동작해야 이중 억제 방지.

---

## 출력 파일

| 파일 | 경로 |
|---|---|
| 결과 영상 | `runs/smoke_detector/video_result_v3/capture1_result_v3.avi` |
| 프레임 로그 | `runs/smoke_detector/video_result_v3/frame_log_v3.csv` |
| 탐지 타임라인 | `runs/smoke_detector/video_result_v3/smoke_timeline_v3.png` |
| 연속성 그래프 | `runs/smoke_detector/video_result_v3/smoke_continuity_v3.png` |

### CSV 로그 컬럼 (v3 추가분 표시)

| 컬럼 | 설명 |
|---|---|
| `smoke_state` | **(v3 신규)** Hysteresis 확정 smoke 상태 |
| `cooldown` | **(v3 신규)** 남은 cooldown 프레임 수 |

---

---

## BalloonGate (흰 기구 오탐 억제)

### 배경

흰 기구가 프레임에 등장하면 두 경로로 오탐 발생:
1. **YOLO 오탐**: 흰 기구가 시각적으로 연기와 유사하게 분류됨
2. **ThinSmoke 오탐**: 흰색 = HSV S채널 ≈ 0 → 프레임 전체 채도 평균 급락 → sat_ratio 조건 충족

### 핵심 아이디어: 연기 vs 흰 기구의 형태 차이

| 특성 | 연기 | 흰 기구 |
|---|---|---|
| 형태 | 불규칙·확산, 경계 흐림 | **원형/타원형, 경계 선명** |
| 원형도 (circularity) | 낮음 (0.1~0.3) | **높음 (0.7~0.9)** |
| 볼록도 (convexity) | 낮음 | **높음 (≈1.0)** |
| 밝기 (HSV V) | 중간 | **극단적으로 높음 (≈255)** |

### 방법 1: BalloonGate 클래스 (흰 볼록체 감지)

```python
# 흰 영역 마스크: V > 200 AND S < 40
white_mask = cv2.inRange(hsv, (0,0,200), (180,40,255))
# 원형도 = 4π × area / perimeter²  (완전한 원 = 1.0)
# 볼록도 = contour면적 / convex hull면적
# 조건 충족 시 is_balloon=True → YOLO vote 및 ThinSmoke 억제
```

파라미터:

| 파라미터 | 값 | 의미 |
|---|---|---|
| `WHITE_V_MIN` | `200` | HSV 밝기 하한 |
| `WHITE_S_MAX` | `40` | HSV 채도 상한 |
| `BLOB_AREA_MIN` | `1000` | 유효 blob 최소 면적 (px) |
| `BLOB_CIRCULARITY_MIN` | `0.45` | 원형도 하한 |
| `BLOB_CONVEXITY_MIN` | `0.80` | 볼록도 하한 |
| `BALLOON_YOLO_SUPPRESS_THR` | `0.65` | YOLO 억제 conf 상한 (이상이면 실제 연기로 간주) |

### 방법 2: ROI 마스킹 (ThinSmoke 특징 계산에서 흰 영역 제외)

```python
# Laplacian 계산 후 유효 픽셀 분산만 추출 (경계 인위적 엣지 방지)
lap = cv2.Laplacian(gray, cv2.CV_64F)
sharpness = float(np.var(lap[valid_pixels]))
# 채도도 흰 영역 제외 픽셀만 평균
saturation = float(hsv[:,:,1][valid_pixels].mean())
```

기구가 없는 상황에서도 항상 적용 → 통계 왜곡 원천 차단

### 오버레이 색상 추가

| 색상 | 의미 |
|---|---|
| 빨간색 | Smoke 탐지 |
| 초록색 | No Smoke |
| **주황색** | **BalloonGate 활성 (흰 기구 감지)** |

---

## 검증 항목

| # | 항목 | 목표 |
|---|---|---|
| 1 | 0~30초 구간 오탐 수 | 0건 유지 |
| 2 | 31초 이후 smoke 구간 끊김 | 최소화 (연속성 유지) |
| 3 | ThinSmoke 발화 프레임 수 | v2(30장) 대비 증가 예상 |
| 4 | 최종 smoke 비율 | v2(44.5%) 대비 소폭 증가 허용 |
| 5 | 처리 속도 | v2(3.3ms/frame) 대비 큰 차이 없어야 함 |
