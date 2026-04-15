# 얇은 연기 연속 탐지 개선 — Claude Code 작업 지시서

> 대상 파일: `src/video_inference_v2.ipynb` (또는 `src/video_inference_v2.py`)  
> 목적: 얇은 연기 구간에서 탐지가 끊기는 문제 해결 (Recall 최우선)  
> 원칙: **오탐(FP)은 허용, 미탐(FN)은 불허**

---

## 현재 문제

얇은 연기가 발생 시점과 중간에 잠깐씩만 탐지되고, 연기가 지속되는 구간에서 탐지가 끊김.
원인: 현재 로직의 진입/해제 조건이 동일하여 경계선 프레임에서 on/off 반복 발생.

---

## 수정 사항 (3단계, 순서대로 적용)

### 1단계 — Cooldown (최소 유지 시간) [필수, 최우선 적용]

smoke 판정이 한 번 발생하면 최소 `COOLDOWN_FRAMES` 동안 smoke 상태를 강제 유지한다.
연기는 물리적으로 순간 사라지지 않으므로, 이 가정은 안전하다.

#### 파라미터

```python
COOLDOWN_FRAMES = 45  # 30fps 기준 약 1.5초
```

#### 구현 위치

최종 판정 로직 (`final_smoke = yolo_vote or thin_smoke`) 직후에 삽입.

#### 코드

```python
# --- 초기화 (추론 루프 시작 전) ---
cooldown_counter = 0

# --- 매 프레임 판정 후 ---
# 기존: final_smoke = yolo_vote or thin_smoke
# 아래를 추가:

if final_smoke:
    cooldown_counter = COOLDOWN_FRAMES
elif cooldown_counter > 0:
    final_smoke = True   # smoke 강제 유지
    cooldown_counter -= 1
```

---

### 2단계 — ThinSmoke 지속성 조건 완화

현재 persist 조건이 너무 까다로워 얇은 연기가 flickering하는 구간에서 탐지 실패.

#### 변경

```python
# 변경 전
PERSIST_WINDOW = 15
PERSIST_MIN_COUNT = 12   # 80%

# 변경 후
PERSIST_WINDOW = 30
PERSIST_MIN_COUNT = 15   # 50%
```

#### 적용 위치

ThinSmoke v2 판정 로직에서 `persist_buf` 관련 파라미터를 위 값으로 교체.

---

### 3단계 — Hysteresis (이력 임계값) [선택 사항]

진입과 해제에 서로 다른 조건을 적용하여 경계선 flickering 제거.

#### 개념

- **진입 조건** (no_smoke → smoke): 현재 기준 그대로 유지
  - `YOLO conf >= 0.40` OR `ThinSmoke v2 조건 충족`
- **해제 조건** (smoke → no_smoke): 훨씬 엄격하게
  - `EXIT_CONSECUTIVE` 프레임 연속으로 아래 조건을 **모두** 만족해야 해제:
    - `YOLO conf < 0.15`
    - `sharp_ratio > 0.90`
    - `sat_ratio > 0.95`

#### 파라미터

```python
EXIT_CONSECUTIVE = 30        # 30프레임 연속 깨끗해야 해제 (약 1초)
EXIT_YOLO_CONF = 0.15        # YOLO confidence 상한
EXIT_SHARP_RATIO = 0.90      # sharpness ratio 하한
EXIT_SAT_RATIO = 0.95        # saturation ratio 하한
```

#### 코드

```python
# --- 초기화 ---
smoke_state = False
clear_streak = 0

# --- 매 프레임 (cooldown 적용 전 위치에서) ---
raw_smoke = yolo_vote or thin_smoke

if not smoke_state:
    # 현재 no_smoke 상태 → 진입 조건 판단
    if raw_smoke:
        smoke_state = True
        clear_streak = 0
else:
    # 현재 smoke 상태 → 해제 조건 판단
    is_clearly_clean = (
        yolo_conf < EXIT_YOLO_CONF
        and sharp_ratio > EXIT_SHARP_RATIO
        and sat_ratio > EXIT_SAT_RATIO
    )
    if is_clearly_clean:
        clear_streak += 1
        if clear_streak >= EXIT_CONSECUTIVE:
            smoke_state = False
            clear_streak = 0
    else:
        clear_streak = 0

final_smoke = smoke_state
# 이후 cooldown 로직 적용
```

---

## 적용 후 검증

### 확인할 항목

1. **0~30초 구간 오탐 수**: 0건 유지 확인 (깨끗한 구간에서 FP 없어야 함)
2. **얇은 연기 구간 연속성**: 31초 이후 연기 구간에서 탐지 끊김 없이 연속 유지되는지 확인
3. **ThinSmoke 발화 프레임 수**: 기존 28장 → 증가 예상
4. **최종 smoke 비율**: 기존 44.5% → 소폭 증가 예상 (허용 범위)
5. **처리 속도**: 기존 3.9 ms/frame 대비 큰 차이 없어야 함

### 검증 출력 추가 (선택)

추론 완료 후 아래 통계를 출력하면 디버깅에 유용:

```python
print(f"Cooldown 강제 유지 프레임 수: {cooldown_forced_count}")
print(f"Hysteresis 해제 거부 프레임 수: {hysteresis_held_count}")
print(f"ThinSmoke 탐지 프레임 수: {thin_smoke_count}")
print(f"최종 smoke 프레임 수: {final_smoke_count} / {total_frames} ({final_smoke_count/total_frames*100:.1f}%)")
```

---

## 주의사항

- **Cooldown(1단계)만 먼저 적용해서 테스트 후**, 부족하면 2단계, 3단계 순서로 추가할 것
- 3단계 Hysteresis 적용 시 cooldown과 함께 사용하면 해제가 이중으로 억제되므로, cooldown은 hysteresis 판정 결과(`smoke_state`) 기준으로 동작하도록 순서 주의
- 파라미터 값(`COOLDOWN_FRAMES`, `PERSIST_WINDOW` 등)은 영상 특성에 따라 조정 필요
