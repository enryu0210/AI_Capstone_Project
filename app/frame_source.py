"""
frame_source.py
===============
영상 입력원(웹캠·캡처카드·비디오 파일) 추상화.

[설계 의도]
- GUI/워커 코드는 입력원의 종류를 알 필요 없게 한다 (Strategy 패턴).
- 모든 소스는 BGR uint8 numpy 배열을 반환하므로 OpenCV/cv2.VideoWriter 와
  자연스럽게 호환된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


class FrameSource:
    """OpenCV VideoCapture 래퍼.

    Parameters
    ----------
    source
        - ``int`` : 카메라/캡처카드 인덱스 (예: 0, 1)
        - ``str`` 또는 ``Path`` : 비디오 파일 경로
    """

    def __init__(self, source: Union[int, str, Path]) -> None:
        # 디버그/UI 표시용으로 원본 식별자 보관
        self._source_repr: Union[int, str] = (
            int(source) if isinstance(source, int) else str(source)
        )

        # cv2.VideoCapture 는 str 또는 int 만 받음. Path 는 변환 필요.
        if isinstance(source, Path):
            cap_arg: Union[int, str] = str(source)
        else:
            cap_arg = source

        # Windows 에서 카메라 백엔드 자동 선택이 가끔 실패하지만, 여기선 기본 선택을
        # 따른다. 향후 문제 시 cv2.CAP_DSHOW 등 명시 백엔드 옵션을 추가할 수 있음.
        self._cap = cv2.VideoCapture(cap_arg)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"입력 소스를 열 수 없습니다: {self._source_repr}\n"
                f"카메라가 연결돼 있는지, 파일 경로가 올바른지 확인하세요."
            )

    # ── 기본 속성 ──────────────────────────────────────────────────
    @property
    def source_repr(self) -> Union[int, str]:
        """UI 표시용 식별자 (예: ``0`` 또는 ``"capture1.avi"``)."""
        return self._source_repr

    def fps(self) -> float:
        """소스 보고 FPS. 카메라는 0 을 반환할 수도 있음 (그 경우 30 으로 가정)."""
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1e-3 else 30.0

    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` 튜플."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    # ── 프레임 획득 ─────────────────────────────────────────────────
    def read(self) -> Optional[np.ndarray]:
        """다음 프레임을 BGR uint8 (H,W,3) numpy 배열로 반환.

        파일 끝, 카메라 분리, 일시적 read 실패 시 ``None``.
        호출자는 이 신호를 보고 루프를 종료하면 된다.
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    # ── 정리 ───────────────────────────────────────────────────────
    def release(self) -> None:
        """캡처 리소스 해제. 같은 인스턴스에 두 번 호출돼도 안전."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    # 컨텍스트 매니저로도 사용 가능 (테스트/스크립트에서 유용)
    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


# ── 카메라 자동 탐지 헬퍼 ──────────────────────────────────────────
def list_available_cameras(max_index: int = 5) -> list[int]:
    """0..max_index-1 의 카메라/캡처카드 인덱스 중 실제로 열리는 번호만 반환.

    GUI 의 Source 콤보박스에 채울 후보를 만들 때 사용.
    매번 호출하면 약간의 지연이 발생하므로 GUI 시작 시 1회만 호출 권장.

    OpenCV 의 UVC 백엔드는 존재하지 않는 인덱스에 대해 "Camera index out of range"
    같은 메시지를 ``stderr`` 에 직접 출력한다. ``cv2.utils.logging.setLogLevel`` 로는
    이 native 출력이 막히지 않으므로, 이 함수에서는 OS 파일 디스크립터 레벨로
    stderr 을 잠시 NUL 로 보낸다 (Python 사이드 stderr 도 같이 막혀, 단 이 짧은
    구간에선 상관없는 메시지가 거의 없다).
    """
    import os
    import sys

    # stderr FD(2) 를 NUL 로 임시 리디렉션
    try:
        original_fd = os.dup(2)
    except OSError:
        original_fd = None  # 환경에 따라 dup 실패 가능 → 노이즈 감수하고 진행

    devnull_fd: Optional[int] = None
    if original_fd is not None:
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
        except OSError:
            # 폴백: 그냥 노이즈 출력
            if devnull_fd is not None:
                os.close(devnull_fd)
            os.close(original_fd)
            original_fd = None
            devnull_fd = None

    available: list[int] = []
    try:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append(idx)
                cap.release()
    finally:
        # 원상복구
        if original_fd is not None:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(original_fd, 2)
            os.close(original_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
    return available
