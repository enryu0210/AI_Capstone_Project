"""
settings.py
===========
앱 설정 영속화 — 마지막으로 사용한 카메라 인덱스, 영상 파일 경로,
저장 폴더 등을 다음 실행에서도 기억하기 위한 얇은 래퍼.

QSettings 는 Windows 에서는 레지스트리, macOS 에서는 plist 등에 자동 저장된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings


# 조직/앱 이름 — QSettings 가 저장 위치를 결정할 때 사용. 한 번 정해지면
# 변경 시 기존 설정이 사라지므로 신중히 정한다.
ORG_NAME = "Capstone"
APP_NAME = "PFAN_Desmoke"


def _settings() -> QSettings:
    return QSettings(ORG_NAME, APP_NAME)


# ── 마지막 카메라 인덱스 ───────────────────────────────────────────
def get_last_camera_index(default: int = 0) -> int:
    val = _settings().value("source/last_camera_index", default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def set_last_camera_index(idx: int) -> None:
    _settings().setValue("source/last_camera_index", int(idx))


# ── 마지막 영상 파일 경로 ──────────────────────────────────────────
def get_last_video_path() -> Optional[Path]:
    val = _settings().value("source/last_video_path", None)
    return Path(val) if val else None


def set_last_video_path(path: Path) -> None:
    _settings().setValue("source/last_video_path", str(path))


# ── 마지막 녹화 저장 폴더 ──────────────────────────────────────────
def get_last_record_dir(default: Optional[Path] = None) -> Path:
    val = _settings().value("record/last_dir", None)
    if val:
        return Path(val)
    if default is not None:
        return default
    return Path.home()


def set_last_record_dir(path: Path) -> None:
    _settings().setValue("record/last_dir", str(path))
