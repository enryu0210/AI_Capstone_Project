"""
inference_worker.py
===================
영상 입력원에서 프레임을 가져와 PFAN 디스모킹 추론을 수행하고,
결과를 메인 GUI 스레드로 전달하는 ``QThread`` 워커.

[왜 별도 스레드인가]
- 추론은 GPU 라도 30~50 ms/frame 소요 → 메인 스레드에서 돌리면 UI 가 멈춤.
- PySide6 의 Signal/Slot 은 thread-safe 이므로 워커→GUI 통신에 안전하다.

[수명주기]
1. 메인 스레드: ``worker = InferenceWorker(engine, source); worker.start()``
2. 워커는 ``run()`` 루프를 돌며 매 프레임 ``frame_processed`` Signal 송출
3. 종료 시: ``worker.request_stop(); worker.wait()``
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from app.desmoke_engine import DesmokeEngine
from app.frame_source import FrameSource


def bgr_to_qimage(bgr: np.ndarray):
    """OpenCV BGR uint8 → PySide6 QImage (RGB888) 변환.

    QImage 는 numpy 버퍼를 그대로 참조하므로 ``copy()`` 로 분리해야
    GC 후에도 안전하게 사용할 수 있다.
    """
    # 지연 임포트 — bgr_to_qimage 만 따로 쓰일 일은 적지만 모듈 단독 테스트 가능하게.
    from PySide6.QtGui import QImage

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # bytesPerLine 명시: numpy 배열이 contiguous 가 아닐 수도 있음을 대비.
    bytes_per_line = rgb.strides[0]
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg.copy()  # 버퍼 독립 사본


class InferenceWorker(QThread):
    """프레임 획득 → 디스모킹 추론 → Signal 송출.

    Signals
    -------
    frame_processed(QImage, QImage, float)
        (원본 RGB QImage, 디스모킹 RGB QImage, latency_ms)
    stats_updated(float, int)
        (현재 FPS, 누적 프레임 수)
    finished_with_reason(str)
        루프가 끝났을 때 사유 문자열 ("end_of_stream", "stopped", "error: ...").
    """

    frame_processed = Signal(object, object, float)   # QImage, QImage, latency_ms
    stats_updated = Signal(float, int)                # fps, frame_index
    finished_with_reason = Signal(str)

    def __init__(
        self,
        engine: DesmokeEngine,
        source: FrameSource,
        *,
        record_path: Optional[Path] = None,
        target_fps: Optional[float] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._source = source
        self._record_path = record_path
        self._target_fps = target_fps

        # 협력적 종료 플래그 — QThread.requestInterruption 도 가능하지만
        # 직접 플래그로 표현하면 의도가 명확.
        self._stop_requested = False

        # FPS 측정용 윈도우 (지연 평균은 EMA, 표시 FPS 는 최근 1초 윈도우)
        self._frame_count = 0
        self._fps_window_start = 0.0
        self._fps_window_count = 0

    # ── 외부에서 호출하는 종료 신호 ─────────────────────────────────
    def request_stop(self) -> None:
        """루프 종료를 요청. ``wait()`` 와 함께 사용."""
        self._stop_requested = True

    # ── QThread 진입점 ──────────────────────────────────────────────
    def run(self) -> None:  # noqa: C901  (조금 길지만 한 흐름이라 분해 안 함)
        writer: Optional[cv2.VideoWriter] = None
        reason = "stopped"
        try:
            # ─ 녹화 활성화 시 VideoWriter 준비 ─────────────────────
            if self._record_path is not None:
                w, h = self._source.resolution()
                # 입력 해상도가 0 이면(일부 카메라) 기본 640x480 으로 폴백
                if w <= 0 or h <= 0:
                    w, h = 640, 480
                fps = self._source.fps()
                # mp4v 코덱 — 대부분의 플레이어가 재생 가능. 코덱 미지원 시 자동 폴백.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(self._record_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    writer = None  # 녹화는 실패하더라도 미리보기는 계속

            # ─ 메인 루프 ───────────────────────────────────────────
            self._fps_window_start = time.perf_counter()
            self._fps_window_count = 0

            while not self._stop_requested:
                frame_bgr = self._source.read()
                if frame_bgr is None:
                    reason = "end_of_stream"
                    break

                t0 = time.perf_counter()
                clean_bgr = self._engine.process_bgr_frame(frame_bgr)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                # 1) 결과 송출 (QImage 변환 후 송출 — Slot 측은 setPixmap 만)
                #    Signal 인자 타입은 object 로 두어 import 의존을 낮춤.
                orig_qimg = bgr_to_qimage(frame_bgr)
                clean_qimg = bgr_to_qimage(clean_bgr)
                self.frame_processed.emit(orig_qimg, clean_qimg, latency_ms)

                # 2) 녹화
                if writer is not None:
                    writer.write(clean_bgr)

                # 3) FPS 측정 — 1초 윈도우
                self._frame_count += 1
                self._fps_window_count += 1
                now = time.perf_counter()
                window_dt = now - self._fps_window_start
                if window_dt >= 1.0:
                    fps = self._fps_window_count / window_dt
                    self.stats_updated.emit(fps, self._frame_count)
                    self._fps_window_start = now
                    self._fps_window_count = 0

                # 4) target_fps 가 지정된 경우 (예: 영상 파일 재생을 원본 속도로)
                if self._target_fps and self._target_fps > 0:
                    expected_dt = 1.0 / self._target_fps
                    spent = time.perf_counter() - t0
                    if spent < expected_dt:
                        # msleep 은 정수 ms — 짧으면 그냥 yield
                        sleep_ms = max(1, int((expected_dt - spent) * 1000))
                        self.msleep(sleep_ms)

        except Exception as e:  # 예측 못한 에러는 사유에 담아 GUI 에 전달
            reason = f"error: {e!r}"
        finally:
            # 항상 리소스 정리
            try:
                if writer is not None:
                    writer.release()
            finally:
                self._source.release()
            self.finished_with_reason.emit(reason)
