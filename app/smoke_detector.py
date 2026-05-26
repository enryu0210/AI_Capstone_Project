"""
smoke_detector.py
=================
복강경 수술 영상용 실시간 연기 탐지기.

[기원]
- 본 코드는 ``feature/detection`` 브랜치의 ``src/video_inference_v3.ipynb`` 에서
  검증된 알고리즘을 PySide6 GUI 환경에 맞게 단일 클래스로 캡슐화한 것이다.

[탐지 파이프라인 (매 프레임)]
    ① YOLOv8n-cls           → smoke confidence (raw)
    ② Sliding Window Vote   → 5프레임 중 4개 이상이면 YOLO smoke 확정
    ③ BalloonGate           → 흰 볼록체(기구) 감지 시
                                · YOLO conf 가 낮으면 vote 결과 억제
                                · ThinSmoke 보조 탐지는 즉시 무효화
                                · white_mask 를 ThinSmoke 특징 계산에서 제외
    ④ ThinSmokeDetectorV2   → 얇은 연기 보조 탐지
                                · Sharpness(30프레임 median) + Saturation 동시 하락
                                · 30프레임 중 15개 이상 지속 시 발화
                                · Rolling 80th percentile 베이스라인 (300프레임)
    ⑤ Hysteresis            → CLEAR→SMOKE 즉시 전환, SMOKE→CLEAR 는 5연속 clean 필요
    ⑥ Cooldown              → smoke 종료 후 15프레임(0.5초) 강제 유지

[GUI 사용 의도]
- ``SmokeDetector.is_available()`` 로 가중치 존재 여부 확인 후 enable.
- ``detector.run(bgr_frame)`` 만 호출하면 ``DetectionResult`` 가 반환됨 — 워커는
  반환값만 UI 시그널로 전달하면 된다.
- ``reset()`` 으로 영상이 바뀌었을 때 내부 버퍼/상태머신 초기화.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────
# 파라미터 — v3 노트북 기본값을 그대로 옮김. 운영 중 튜닝이 필요하면
# 한 곳에서 수정할 수 있도록 상수 블록으로 분리.
# ──────────────────────────────────────────────────────────────────

# YOLO 분류기
SMOKE_THR = 0.40       # raw confidence 임계값 — 얇은 연기 포함을 위해 낮춘 값
VOTE_WINDOW = 5
VOTE_K = 4             # v3 에서 3 → 4 로 강화: 단발 FP 추가 억제

# ThinSmoke v2
WARMUP_FRAMES = 300                # 베이스라인 워밍업 (≈ 10초 @ 30fps)
BASELINE_WINDOW = 300
BASELINE_PCT = 80                  # 80th percentile = "선명한 정상" 추적
SHARP_SMOOTH_WINDOW = 30           # 30프레임 rolling median (카메라 이동 노이즈 흡수)
SHARP_DROP_RATIO = 0.50
SAT_DROP_RATIO = 0.82
YOLO_UNCERTAIN_LO = 0.00           # 하한 제거: 이중 게이트가 보호 담당
YOLO_UNCERTAIN_HI = 0.40           # YOLO 가 확신(>0.40)이면 보조 탐지 불필요
PERSIST_WINDOW = 30                # v3: 15 → 30
PERSIST_THRESH = 15                # v3: 12 → 15 (창이 2배여서 비율은 비슷)

# Hysteresis / Cooldown
EXIT_CONSECUTIVE = 5               # v3: 15 → 5 (상태 해제 빠르게)
COOLDOWN_FRAMES = 15               # v3: 45 → 15 (강제유지 축소)

# BalloonGate (흰 기구 오탐 억제)
WHITE_V_MIN = 200
WHITE_S_MAX = 40
BLOB_AREA_MIN = 1000
BLOB_CIRCULARITY_MIN = 0.45
BLOB_CONVEXITY_MIN = 0.80
BALLOON_YOLO_SUPPRESS_THR = 0.65   # 이 신뢰도 이상이면 기구 옆 진짜 연기로 보고 억제 안함


# ──────────────────────────────────────────────────────────────────
# 가중치 위치 탐색
# ──────────────────────────────────────────────────────────────────
def _default_weight_candidates() -> list[Path]:
    """기본 가중치 후보 경로들. 앞쪽에 있을수록 우선."""
    here = Path(__file__).resolve().parent.parent
    return [
        # detection 브랜치 학습 결과 표준 경로
        here / "runs" / "smoke_detector" / "yolov8n_cls" / "weights" / "best.pt",
        # 보조 — 클라이언트 패키지 내부에 직접 떨어뜨려도 동작
        here / "app" / "weights" / "smoke_best.pt",
        here / "weights" / "smoke_best.pt",
    ]


def find_weights(explicit: Optional[Path] = None) -> Optional[Path]:
    """우선순위: explicit > 환경변수 SMOKE_DETECTOR_WEIGHTS > 기본 후보."""
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)

    env = os.environ.get("SMOKE_DETECTOR_WEIGHTS")
    if env and Path(env).exists():
        return Path(env)

    for p in _default_weight_candidates():
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────────
# 결과 데이터 컨테이너
# ──────────────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    """한 프레임의 탐지 결과.

    UI / 워커 모두 이 객체 하나만 보고 화면을 갱신하면 된다.
    """
    is_smoke: bool                          # 최종 판정 (cooldown / hysteresis 적용 후)
    yolo_conf: float                        # YOLO smoke confidence (raw)
    yolo_raw: bool                          # raw 임계값 통과 여부
    yolo_vote: bool                         # sliding vote 통과 여부 (balloon 억제 반영)
    vote_count: int                         # 최근 5프레임 중 smoke 수 (0-5)
    thin_smoke: bool                        # ThinSmoke v2 발화 여부
    is_balloon: bool                        # BalloonGate 감지 여부
    smoke_state: bool                       # Hysteresis 상태 머신의 현재값
    cooldown_remaining: int                 # 잔여 cooldown 프레임 수
    persist_count: int                      # 최근 30프레임 중 이상 신호 수
    extra: dict = field(default_factory=dict)  # 디버그용 (sharp_ratio 등)


# ──────────────────────────────────────────────────────────────────
# Rolling Percentile Baseline
# ──────────────────────────────────────────────────────────────────
class _RollingPercentileBaseline:
    """최근 N프레임의 P 분위수를 베이스라인으로 사용.

    EMA 대비 장점은 초반 고변동 구간에서 오염되지 않는다는 점.
    버퍼가 절반 이상 차야 의미 있는 값을 반환한다.
    """

    def __init__(self, window: int = 300, percentile: int = 80) -> None:
        self.buf: deque[float] = deque(maxlen=window)
        self.percentile = percentile

    def update(self, value: float) -> None:
        self.buf.append(value)

    def get_baseline(self) -> Optional[float]:
        maxlen = self.buf.maxlen or 1
        if len(self.buf) < maxlen // 2:
            return None
        return float(np.percentile(list(self.buf), self.percentile))

    def reset(self) -> None:
        self.buf.clear()


# ──────────────────────────────────────────────────────────────────
# BalloonGate — 흰 볼록체(기구) 감지
# ──────────────────────────────────────────────────────────────────
class _BalloonGate:
    """흰 기구/볼록체 감지 게이트.

    연기와 흰 기구는 RGB 통계가 비슷해 보일 수 있지만, 윤곽 기하 특성이 다르다.
    - 흰 기구 : circularity / convexity 모두 높음
    - 연기    : 둘 다 낮음 (불규칙·확산)
    """

    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def check(self, frame: np.ndarray) -> tuple[bool, np.ndarray, dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            (0, 0, WHITE_V_MIN),
            (180, WHITE_S_MAX, 255),
        )
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, self._kernel)

        info = {
            "blob_count": 0,
            "max_area": 0,
            "max_circularity": 0.0,
            "max_convexity": 0.0,
        }

        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        is_balloon = False
        best_circ = 0.0
        best_conv = 0.0
        valid_blobs = 0
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < BLOB_AREA_MIN:
                continue
            valid_blobs += 1
            max_area = max(max_area, area)

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter ** 2)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 1.0 else 0.0

            best_circ = max(best_circ, circularity)
            best_conv = max(best_conv, convexity)

            if circularity >= BLOB_CIRCULARITY_MIN and convexity >= BLOB_CONVEXITY_MIN:
                is_balloon = True

        info.update({
            "blob_count": valid_blobs,
            "max_area": int(max_area),
            "max_circularity": round(best_circ, 3),
            "max_convexity": round(best_conv, 3),
        })
        return is_balloon, white_mask, info


# ──────────────────────────────────────────────────────────────────
# 특징 추출 (ROI 마스킹 지원)
# ──────────────────────────────────────────────────────────────────
def _extract_features(
    frame: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Laplacian Sharpness + HSV Saturation.

    ``exclude_mask`` (255=제외) 가 있으면 해당 픽셀을 빼고 통계를 낸다 —
    흰 기구가 sat/sharp 통계를 왜곡하지 않게 하기 위함.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if exclude_mask is not None and exclude_mask.any():
        valid = exclude_mask == 0
        # Laplacian 은 전체에 적용 후 valid 픽셀의 분산만 사용 — 마스킹 전에 미리 빼면
        # 인위적 경계 엣지가 생겨 분산이 부풀려진다.
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        valid_lap = lap[valid]
        sharpness = float(np.var(valid_lap)) if valid_lap.size > 100 else 0.0
        valid_sat = hsv[:, :, 1][valid]
        saturation = float(valid_sat.mean()) if valid_sat.size > 100 else 0.0
    else:
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        saturation = float(hsv[:, :, 1].mean())

    return sharpness, saturation


# ──────────────────────────────────────────────────────────────────
# ThinSmoke v2
# ──────────────────────────────────────────────────────────────────
class _ThinSmokeDetectorV2:
    """얇은 연기 보조 탐지 — Sharpness × Saturation 이중 게이트 + 지속성."""

    def __init__(self) -> None:
        self.sharp_baseline = _RollingPercentileBaseline(BASELINE_WINDOW, BASELINE_PCT)
        self.sat_baseline = _RollingPercentileBaseline(BASELINE_WINDOW, BASELINE_PCT)
        self.persist_buf: deque[int] = deque(maxlen=PERSIST_WINDOW)
        self.sharp_smooth_buf: deque[float] = deque(maxlen=SHARP_SMOOTH_WINDOW)
        self.frame_count = 0

    def reset(self) -> None:
        self.sharp_baseline.reset()
        self.sat_baseline.reset()
        self.persist_buf.clear()
        self.sharp_smooth_buf.clear()
        self.frame_count = 0

    def update(
        self,
        frame: np.ndarray,
        yolo_conf: float,
        white_mask: Optional[np.ndarray],
        is_balloon: bool,
    ) -> tuple[bool, dict]:
        self.frame_count += 1

        sharpness, saturation = _extract_features(frame, exclude_mask=white_mask)
        self.sharp_smooth_buf.append(sharpness)
        self.sharp_baseline.update(sharpness)
        self.sat_baseline.update(saturation)

        smooth_sharpness = float(np.median(list(self.sharp_smooth_buf)))

        info = {
            "sharpness": smooth_sharpness,
            "saturation": saturation,
            "sharp_base": None,
            "sat_base": None,
            "sharp_ratio": None,
            "sat_ratio": None,
            "both_low": False,
            "persist_cnt": 0,
            "suppressed_by_gate": is_balloon,
        }

        if self.frame_count < WARMUP_FRAMES:
            self.persist_buf.append(0)
            return False, info

        if is_balloon:
            # BalloonGate 발동 — 본 프레임의 sat/sharp 하락은 기구에 의한 것일 수 있음.
            # persist 버퍼를 0으로 채워 다음 프레임에 즉시 다시 떨어지지 않도록 한다.
            self.persist_buf.append(0)
            return False, info

        if not (YOLO_UNCERTAIN_LO <= yolo_conf <= YOLO_UNCERTAIN_HI):
            self.persist_buf.append(0)
            return False, info

        sharp_base = self.sharp_baseline.get_baseline()
        sat_base = self.sat_baseline.get_baseline()
        if sharp_base is None or sat_base is None or sharp_base < 1.0 or sat_base < 1.0:
            self.persist_buf.append(0)
            return False, info

        sharp_ratio = smooth_sharpness / sharp_base
        sat_ratio = saturation / sat_base
        both_low = (sharp_ratio < SHARP_DROP_RATIO) and (sat_ratio < SAT_DROP_RATIO)

        info.update({
            "sharp_base": sharp_base,
            "sat_base": sat_base,
            "sharp_ratio": sharp_ratio,
            "sat_ratio": sat_ratio,
            "both_low": both_low,
        })

        self.persist_buf.append(1 if both_low else 0)
        persist_cnt = sum(self.persist_buf)
        info["persist_cnt"] = persist_cnt

        thin = (
            len(self.persist_buf) == PERSIST_WINDOW
            and persist_cnt >= PERSIST_THRESH
        )
        return thin, info


# ──────────────────────────────────────────────────────────────────
# SmokeDetector (GUI 진입점)
# ──────────────────────────────────────────────────────────────────
class SmokeDetector:
    """가중치가 있으면 즉시 동작하는 실시간 연기 탐지기.

    Notes
    -----
    - YOLO 추론은 매 프레임 ``imgsz=224`` 로 진행 (학습 셋업 동일).
    - 매 프레임 1회만 호출되도록 설계됨 — 외부에서 thread-safe 보장은 필요 없음.
    - 가중치 없거나 ultralytics import 실패 시에는 ``is_available() == False``.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        *,
        device: Optional[str] = None,
        imgsz: int = 224,
    ) -> None:
        self._imgsz = imgsz
        self._device = device
        self._model = None
        self._smoke_idx: Optional[int] = None
        self._weights_path: Optional[Path] = None
        self._init_error: Optional[str] = None

        # 내부 상태
        self._vote_buf: deque[int] = deque(maxlen=VOTE_WINDOW)
        self._thin = _ThinSmokeDetectorV2()
        self._gate = _BalloonGate()
        self._smoke_state = False
        self._clear_streak = 0
        self._cooldown = 0
        self._frame_idx = 0

        self._load(weights_path)

    # ─── 초기화 / 가용성 ──────────────────────────────────────
    def _load(self, weights_path: Optional[Path]) -> None:
        path = find_weights(weights_path)
        if path is None:
            self._init_error = "가중치 파일을 찾지 못함 (best.pt)"
            return
        try:
            # ultralytics 는 import 비용이 큰 라이브러리라 lazy import.
            from ultralytics import YOLO
            model = YOLO(str(path))
            # 분류 헤드 라벨에서 smoke index 찾기
            names = model.names
            idx_candidates = [k for k, v in names.items() if str(v).lower() == "smoke"]
            if not idx_candidates:
                self._init_error = f"라벨에 'smoke' 없음: {names}"
                return
            self._model = model
            self._smoke_idx = int(idx_candidates[0])
            self._weights_path = path
        except Exception as e:  # 광범위 — 가중치 깨짐, CUDA 미스매치, ultralytics 미설치 모두 잡음
            self._init_error = f"YOLO 로드 실패: {e!r}"

    def is_available(self) -> bool:
        return self._model is not None

    @property
    def weights_path(self) -> Optional[Path]:
        return self._weights_path

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    # ─── 외부 인터페이스 ─────────────────────────────────────
    def reset(self) -> None:
        """영상 소스가 바뀌었을 때 호출 — 내부 누적 상태를 초기화."""
        self._vote_buf.clear()
        self._thin.reset()
        self._smoke_state = False
        self._clear_streak = 0
        self._cooldown = 0
        self._frame_idx = 0

    def run(self, bgr: np.ndarray) -> DetectionResult:
        """한 프레임을 흘려보내고 ``DetectionResult`` 반환.

        가중치가 없으면 항상 ``is_smoke=False`` 인 더미 결과를 돌려준다 —
        호출 측에서 ``is_available()`` 로 분기해도 되고, 그대로 무시해도 안전.
        """
        self._frame_idx += 1

        if self._model is None or self._smoke_idx is None:
            return DetectionResult(
                is_smoke=False,
                yolo_conf=0.0,
                yolo_raw=False,
                yolo_vote=False,
                vote_count=0,
                thin_smoke=False,
                is_balloon=False,
                smoke_state=False,
                cooldown_remaining=0,
                persist_count=0,
                extra={"reason": "detector unavailable"},
            )

        # ① YOLO 추론
        results = self._model.predict(
            bgr,
            verbose=False,
            imgsz=self._imgsz,
            device=self._device,
        )
        yolo_conf = float(results[0].probs.data[self._smoke_idx])
        yolo_raw = yolo_conf >= SMOKE_THR

        # ② Sliding window vote
        self._vote_buf.append(1 if yolo_raw else 0)
        vote_cnt = sum(self._vote_buf)
        yolo_vote = vote_cnt >= VOTE_K

        # ③ BalloonGate
        is_balloon, white_mask, gate_info = self._gate.check(bgr)
        if is_balloon and yolo_vote and yolo_conf < BALLOON_YOLO_SUPPRESS_THR:
            # 흰 기구 옆의 약한 신호는 가짜일 가능성이 큼 — vote 결과 억제
            yolo_vote = False

        # ④ ThinSmoke 보조 탐지
        thin_smoke, thin_info = self._thin.update(
            bgr, yolo_conf, white_mask if is_balloon else None, is_balloon,
        )
        # white_mask 자체는 BalloonGate 가 ON 일 때만 ROI 마스킹에 사용한다 —
        # 평상 시에도 마스킹하면 너무 강한 노이즈가 들어가 베이스라인이 흔들림.

        # ⑤ Hysteresis 상태 머신
        raw_smoke = yolo_vote or thin_smoke
        if not self._smoke_state:
            # CLEAR → SMOKE 는 워밍업 이후 raw_smoke 가 한 번만 떠도 즉시 전환
            if raw_smoke and self._frame_idx >= WARMUP_FRAMES:
                self._smoke_state = True
                self._clear_streak = 0
        else:
            if not raw_smoke:
                self._clear_streak += 1
                if self._clear_streak >= EXIT_CONSECUTIVE:
                    self._smoke_state = False
                    self._clear_streak = 0
            else:
                self._clear_streak = 0

        # ⑥ Cooldown
        final_smoke = self._smoke_state
        if final_smoke:
            self._cooldown = COOLDOWN_FRAMES
        elif self._cooldown > 0:
            final_smoke = True
            self._cooldown -= 1

        extra = {
            "sharp_ratio": thin_info.get("sharp_ratio"),
            "sat_ratio": thin_info.get("sat_ratio"),
            "blob_circularity": gate_info["max_circularity"],
            "blob_convexity": gate_info["max_convexity"],
            "blob_area": gate_info["max_area"],
        }

        return DetectionResult(
            is_smoke=final_smoke,
            yolo_conf=yolo_conf,
            yolo_raw=yolo_raw,
            yolo_vote=yolo_vote,
            vote_count=vote_cnt,
            thin_smoke=thin_smoke,
            is_balloon=is_balloon,
            smoke_state=self._smoke_state,
            cooldown_remaining=self._cooldown,
            persist_count=thin_info.get("persist_cnt", 0),
            extra=extra,
        )
