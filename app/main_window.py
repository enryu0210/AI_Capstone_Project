"""
main_window.py
==============
PFAN 디스모킹 PC 클라이언트의 메인 윈도우.

[레이아웃]
    ┌─────────────────────────────────────────────────────────────┐
    │ Source: [▾]  Device: [▾]  [Start] [Stop] [Snapshot] [Record]│
    ├──────────────────────────┬──────────────────────────────────┤
    │      원본 미리보기       │       디스모킹 결과              │
    │   (QLabel, 16:9 유지)    │     (QLabel, 16:9 유지)          │
    ├──────────────────────────┴──────────────────────────────────┤
    │ Status: idle │ FPS: -- │ Latency: -- ms │ Frames: 0          │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app import settings as app_settings
from app.desmoke_engine import DesmokeEngine
from app.frame_source import FrameSource, list_available_cameras
from app.inference_worker import InferenceWorker, bgr_to_qimage  # noqa: F401


# 비디오 파일 선택용 가상 인덱스. 콤보박스에서 음수 값을 사용해 카메라 인덱스와 충돌 방지.
SOURCE_VIDEO_FILE = -1


class PreviewLabel(QLabel):
    """16:9 비율을 유지하면서 부모 위젯 크기에 맞춰 스케일되는 미리보기 라벨."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName(title)
        # 검은 배경 + 흰 텍스트로 영상 미연결 상태도 명확히 보이게.
        self.setStyleSheet(
            "QLabel { background-color: #111; color: #ccc; "
            "border: 1px solid #333; }"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(f"{title}\n(Source 선택 후 Start)")
        self.setMinimumSize(320, 180)

    def sizeHint(self) -> QSize:
        return QSize(640, 360)

    def update_image(self, qimg: QImage) -> None:
        """들어온 QImage 를 라벨 크기에 맞춰 스케일하여 표시."""
        # KeepAspectRatio + SmoothTransformation: 비율 보존 + 부드러운 보간
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PFAN De-Smoking — 실시간 미리보기")
        self.resize(1280, 720)

        # ── 상태 ────────────────────────────────────────────────
        self._engine: Optional[DesmokeEngine] = None
        self._worker: Optional[InferenceWorker] = None
        self._latest_clean_qimg: Optional[QImage] = None  # 스냅샷 저장용
        self._record_active: bool = False                  # 다음 Start 가 녹화도 할지 여부

        # ── 레이아웃 구성 ───────────────────────────────────────
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 1) 상단 컨트롤 바
        controls = QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(220)
        controls.addWidget(self.source_combo)

        controls.addSpacing(12)
        controls.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU (cuda:0)", "cuda:0")
        self.device_combo.addItem("CPU", "cpu")
        controls.addWidget(self.device_combo)

        controls.addStretch(1)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_snapshot = QPushButton("Snapshot")
        self.btn_record = QPushButton("● Record")
        self.btn_record.setCheckable(True)
        for btn in (self.btn_start, self.btn_stop, self.btn_snapshot, self.btn_record):
            btn.setMinimumWidth(96)
            controls.addWidget(btn)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

        root.addLayout(controls)

        # 2) 두 미리보기 영역
        previews = QHBoxLayout()
        previews.setSpacing(8)
        self.preview_orig = PreviewLabel("원본")
        self.preview_clean = PreviewLabel("디스모킹 결과")
        previews.addWidget(self.preview_orig, 1)
        previews.addWidget(self.preview_clean, 1)
        root.addLayout(previews, 1)

        # 3) 상태바
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self._lbl_state = QLabel("idle")
        self._lbl_fps = QLabel("FPS: --")
        self._lbl_latency = QLabel("Latency: -- ms")
        self._lbl_frames = QLabel("Frames: 0")
        for lbl in (self._lbl_state, self._lbl_fps, self._lbl_latency, self._lbl_frames):
            lbl.setMinimumWidth(110)
            self.status.addPermanentWidget(lbl)

        # ── 시그널 연결 ─────────────────────────────────────────
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        self.btn_record.toggled.connect(self._on_record_toggle)

        # ── Source 콤보 채우기 ──────────────────────────────────
        self._populate_source_combo()

    # ── Source 콤보 구성 ────────────────────────────────────────
    def _populate_source_combo(self) -> None:
        self.source_combo.clear()
        cams = list_available_cameras(max_index=4)
        for idx in cams:
            self.source_combo.addItem(f"Camera {idx}", idx)
        # "비디오 파일..." 항목은 마지막에 별도로 추가
        self.source_combo.addItem("비디오 파일...", SOURCE_VIDEO_FILE)

        # 마지막 사용 카메라가 살아있으면 그걸 선택
        last_cam = app_settings.get_last_camera_index()
        for i in range(self.source_combo.count()):
            if self.source_combo.itemData(i) == last_cam:
                self.source_combo.setCurrentIndex(i)
                break

    # ── Source 결정 (콤보 선택 → FrameSource 인스턴스) ──────────
    def _resolve_source(self) -> Optional[FrameSource]:
        data = self.source_combo.currentData()
        if data == SOURCE_VIDEO_FILE:
            # 파일 다이얼로그
            last = app_settings.get_last_video_path()
            start_dir = str(last.parent) if last else str(Path.cwd())
            path_str, _ = QFileDialog.getOpenFileName(
                self,
                "비디오 파일 선택",
                start_dir,
                "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
            )
            if not path_str:
                return None
            path = Path(path_str)
            app_settings.set_last_video_path(path)
            try:
                return FrameSource(path)
            except RuntimeError as e:
                QMessageBox.warning(self, "오류", str(e))
                return None
        else:
            cam_idx = int(data)
            try:
                src = FrameSource(cam_idx)
            except RuntimeError as e:
                QMessageBox.warning(self, "오류", str(e))
                return None
            app_settings.set_last_camera_index(cam_idx)
            return src

    # ── Start / Stop ────────────────────────────────────────────
    def _on_start(self) -> None:
        # 1) 엔진 준비 (없거나 device 가 바뀐 경우 새로 만든다)
        target_device = self.device_combo.currentData()
        if self._engine is None or str(self._engine.device) != target_device:
            try:
                self._set_state(f"Loading model ({target_device})...")
                QApplication.processEvents()
                self._engine = DesmokeEngine.from_default(device=target_device)
            except Exception as e:
                self._set_state("error")
                QMessageBox.critical(self, "모델 로드 실패", f"{e}")
                return

        # 2) 소스 결정
        source = self._resolve_source()
        if source is None:
            self._set_state("idle")
            return

        # 3) 녹화 경로 결정 (체크돼 있으면)
        record_path: Optional[Path] = None
        if self._record_active:
            record_path = self._ask_record_path()
            if record_path is None:
                source.release()
                self.btn_record.setChecked(False)  # 사용자 취소 시 토글 해제
                self._record_active = False
                return

        # 4) 비디오 파일 재생일 땐 원본 속도 유지 (camera 는 자체 속도)
        target_fps = None
        if isinstance(source.source_repr, str):  # 파일 경로
            target_fps = source.fps()

        # 5) 워커 생성/시작
        self._worker = InferenceWorker(
            engine=self._engine,
            source=source,
            record_path=record_path,
            target_fps=target_fps,
        )
        self._worker.frame_processed.connect(self._on_frame)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.finished_with_reason.connect(self._on_worker_finished)
        self._worker.start()

        # 6) UI 상태 전환
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.btn_record.setEnabled(False)  # 진행 중에는 녹화 토글 변경 금지
        self._set_state("running" + (" + recording" if record_path else ""))
        self._lbl_frames.setText("Frames: 0")

    def _on_stop(self) -> None:
        if self._worker is None:
            return
        self.btn_stop.setEnabled(False)
        self._set_state("stopping...")
        self._worker.request_stop()
        # wait 는 짧게만 — finished_with_reason 시그널이 와서 정리한다.
        self._worker.wait(3000)

    def _ask_record_path(self) -> Optional[Path]:
        """녹화 저장 경로를 사용자에게 묻는다."""
        default_dir = app_settings.get_last_record_dir(default=Path.cwd())
        suggested = default_dir / f"desmoke_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "녹화 저장 위치",
            str(suggested),
            "MP4 (*.mp4)",
        )
        if not path_str:
            return None
        path = Path(path_str)
        app_settings.set_last_record_dir(path.parent)
        return path

    # ── Snapshot ────────────────────────────────────────────────
    def _on_snapshot(self) -> None:
        if self._latest_clean_qimg is None:
            QMessageBox.information(self, "스냅샷", "아직 처리된 프레임이 없습니다.")
            return
        default_dir = app_settings.get_last_record_dir(default=Path.cwd())
        suggested = default_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "스냅샷 저장", str(suggested), "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path_str:
            return
        path = Path(path_str)
        app_settings.set_last_record_dir(path.parent)
        # QImage.save() 는 한글 경로도 안전하게 처리.
        if not self._latest_clean_qimg.save(str(path)):
            QMessageBox.warning(self, "오류", f"저장 실패: {path}")

    # ── Record 토글 ─────────────────────────────────────────────
    def _on_record_toggle(self, checked: bool) -> None:
        # 녹화 활성 여부는 다음 Start 클릭 시점에 반영. 즉시 시작은 하지 않음
        # (현재 워커가 도는 도중에 갑자기 writer 를 끼워넣지 않게 단순화).
        self._record_active = checked
        self.btn_record.setText("● Recording" if checked else "● Record")

    # ── Worker Slot ────────────────────────────────────────────
    def _on_frame(self, orig_qimg, clean_qimg, latency_ms: float) -> None:
        # PySide6 Signal 의 object 인자는 정확한 타입이 와야 함 (QImage 그대로 전달).
        self.preview_orig.update_image(orig_qimg)
        self.preview_clean.update_image(clean_qimg)
        self._latest_clean_qimg = clean_qimg
        self._lbl_latency.setText(f"Latency: {latency_ms:.0f} ms")

    def _on_stats(self, fps: float, frame_index: int) -> None:
        self._lbl_fps.setText(f"FPS: {fps:.1f}")
        self._lbl_frames.setText(f"Frames: {frame_index}")

    def _on_worker_finished(self, reason: str) -> None:
        # 워커 정리 + UI 복귀
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.btn_record.setEnabled(True)
        self._set_state(f"finished ({reason})")
        # 영상 파일 끝에 도달했다면 안내
        if reason == "end_of_stream":
            QMessageBox.information(self, "완료", "재생이 끝났습니다.")
        elif reason.startswith("error"):
            QMessageBox.warning(self, "처리 중 오류", reason)

    # ── 상태바 보조 ─────────────────────────────────────────────
    def _set_state(self, text: str) -> None:
        self._lbl_state.setText(text)

    # ── 종료 처리 ───────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 시그니처)
        # 워커가 살아있으면 안전하게 정리하고 종료
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(3000)
        super().closeEvent(event)
