"""
main_window.py
==============
PFAN 디스모킹 PC 클라이언트의 메인 윈도우 (다크 테마 + 카드 레이아웃 버전).

[레이아웃 개요]
    ┌─ Header card (둥근 모서리, 어두운 표면) ───────────────────┐
    │ INPUT | DEVICE | ACTIONS                                  │
    │ [📷Camera▾]  [⚡GPU▾]   [▶ Start][⏹ Stop][⏺ Record][📷Snap]│
    └────────────────────────────────────────────────────────────┘
    ┌─ Original ────────────┐  ┌─ Desmoked ───────────┐
    │ ◉ ORIGINAL            │  │ ✓ DESMOKED           │
    │  (영상 표시 영역)      │  │  (영상 표시 영역)     │
    └────────────────────────┘  └────────────────────────┘
    [● running] [FPS 14.2]  [Latency 67ms]  [Frames 1247]
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app import settings as app_settings
from app.desmoke_engine import DesmokeEngine
from app.frame_source import FrameSource, list_available_cameras
from app.inference_worker import InferenceWorker
from app.style import GLOBAL_QSS, latency_level


# ── 콤보박스 데이터 sentinel ────────────────────────────────────
# - int (>= 0)  : 카메라 인덱스
# - "PICK_FILE" : 파일 다이얼로그를 띄우는 트리거 항목
# - 문자열 path : 사용자가 선택한 비디오 파일 경로 (실제 소스)
PICK_FILE_SENTINEL = "PICK_FILE"


class PreviewCard(QFrame):
    """제목 스트립 + 영상 표시 영역으로 구성된 카드형 프리뷰 위젯."""

    def __init__(self, title: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("previewCard")
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 제목 스트립
        self._title_label = QLabel(title.upper())
        self._title_label.setObjectName("previewTitle")
        layout.addWidget(self._title_label)

        # 영상 표시 영역
        self._body = QLabel(subtitle)
        self._body.setObjectName("previewBody")
        self._body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._body.setMinimumSize(360, 240)
        layout.addWidget(self._body, 1)

    def sizeHint(self) -> QSize:
        return QSize(640, 380)

    def update_image(self, qimg: QImage) -> None:
        """들어온 QImage 를 라벨 크기에 맞춰 부드럽게 스케일."""
        pix = QPixmap.fromImage(qimg).scaled(
            self._body.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._body.setPixmap(pix)

    def reset(self, message: str) -> None:
        """프리뷰 영역을 텍스트 안내로 초기화."""
        self._body.clear()
        self._body.setText(message)


class StatePill(QLabel):
    """상태바 좌측의 상태 표시 — running/recording/error 등 동적으로 색이 바뀜."""

    def __init__(self, parent=None) -> None:
        super().__init__("● idle", parent)
        self.setProperty("class", "pillState")
        self.setObjectName("pillState")

    def set_idle(self, text: str = "idle") -> None:
        self._reset_flags()
        self.setText(f"● {text}")
        self._refresh_style()

    def set_live(self, recording: bool) -> None:
        self._reset_flags()
        if recording:
            self.setProperty("recording", "true")
            self.setText("⏺ REC + LIVE")
        else:
            self.setProperty("live", "true")
            self.setText("● LIVE")
        self._refresh_style()

    def set_error(self, text: str) -> None:
        self._reset_flags()
        self.setProperty("error", "true")
        self.setText(f"⚠ {text}")
        self._refresh_style()

    def _reset_flags(self) -> None:
        for k in ("live", "recording", "error"):
            self.setProperty(k, "")

    def _refresh_style(self) -> None:
        # QSS property 변경 후 강제 재적용
        self.style().unpolish(self)
        self.style().polish(self)


class _Pill(QLabel):
    """일반 정보 pill (FPS, Latency, Frames)."""

    def __init__(self, text: str, object_name: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setProperty("class", "pill")
        if object_name:
            self.setObjectName(object_name)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PFAN De-Smoking — 실시간 미리보기")
        self.resize(1320, 760)
        self.setStyleSheet(GLOBAL_QSS)

        # ── 상태 ────────────────────────────────────────────────
        self._engine: Optional[DesmokeEngine] = None
        self._worker: Optional[InferenceWorker] = None
        self._latest_clean_qimg: Optional[QImage] = None
        self._record_active: bool = False
        # 콤보 selection 변경을 사용자 액션과 프로그래매틱 변경을 구분하기 위함
        self._suppress_combo_event: bool = False

        # ── 중앙 위젯 / 루트 레이아웃 ──────────────────────────
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 8)
        root.setSpacing(12)

        root.addWidget(self._build_header())
        root.addLayout(self._build_previews(), 1)

        self.setStatusBar(self._build_status_bar())

        # ── 시그널 ──────────────────────────────────────────────
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        self.btn_record.toggled.connect(self._on_record_toggle)
        # ★ 즉시 파일 다이얼로그: activated 는 사용자 인터랙션에만 발화
        self.source_combo.activated.connect(self._on_source_activated)

        # ── Source 콤보 채우기 ─────────────────────────────────
        self._populate_source_combo()

    # ─────────────────────────────────────────────────────────────
    # 빌더 메서드
    # ─────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        """상단 컨트롤 카드 (Source / Device / Actions)."""
        card = QFrame()
        card.setObjectName("headerCard")

        outer = QHBoxLayout(card)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        # Source 그룹
        outer.addLayout(self._build_source_group())
        outer.addWidget(self._make_divider())

        # Device 그룹
        outer.addLayout(self._build_device_group())
        outer.addWidget(self._make_divider())

        # Actions 그룹 (오른쪽으로 밀기)
        outer.addStretch(1)
        outer.addLayout(self._build_actions_group())

        return card

    def _build_source_group(self) -> QVBoxLayout:
        wrap = QVBoxLayout()
        wrap.setSpacing(4)
        title = QLabel("INPUT")
        title.setObjectName("sectionLabel")
        wrap.addWidget(title)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(280)
        wrap.addWidget(self.source_combo)
        return wrap

    def _build_device_group(self) -> QVBoxLayout:
        wrap = QVBoxLayout()
        wrap.setSpacing(4)
        title = QLabel("DEVICE")
        title.setObjectName("sectionLabel")
        wrap.addWidget(title)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(180)
        self.device_combo.addItem("⚡ GPU (cuda:0)", "cuda:0")
        self.device_combo.addItem("🖥 CPU", "cpu")
        wrap.addWidget(self.device_combo)
        return wrap

    def _build_actions_group(self) -> QVBoxLayout:
        wrap = QVBoxLayout()
        wrap.setSpacing(4)
        title = QLabel("ACTIONS")
        title.setObjectName("sectionLabel")
        wrap.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_record = QPushButton("⏺  Record")
        self.btn_record.setObjectName("btnRecord")
        self.btn_record.setCheckable(True)
        self.btn_snapshot = QPushButton("📷  Snapshot")
        self.btn_snapshot.setObjectName("btnSnapshot")

        for btn in (self.btn_start, self.btn_stop, self.btn_record, self.btn_snapshot):
            btn.setMinimumWidth(108)
            row.addWidget(btn)

        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

        wrap.addLayout(row)
        return wrap

    def _build_previews(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(12)

        self.preview_orig = PreviewCard("◉ Original", "Source 선택 후 ▶ Start 를 누르세요")
        self.preview_clean = PreviewCard("✓ Desmoked", "디스모킹 결과가 여기에 표시됩니다")
        layout.addWidget(self.preview_orig, 1)
        layout.addWidget(self.preview_clean, 1)
        return layout

    def _build_status_bar(self) -> QStatusBar:
        bar = QStatusBar(self)
        bar.setSizeGripEnabled(False)

        # 좌측: 상태 pill
        self._pill_state = StatePill()
        self._pill_state.set_idle()
        bar.addWidget(self._pill_state)

        # 우측 (영구 위젯): FPS, Latency, Frames
        self._pill_fps = _Pill("FPS  --", "pillFps")
        self._pill_latency = _Pill("Latency  --  ms", "pillLatency")
        self._pill_frames = _Pill("Frames  0", "pillFrames")
        for p in (self._pill_fps, self._pill_latency, self._pill_frames):
            bar.addPermanentWidget(p)

        return bar

    def _make_divider(self) -> QFrame:
        div = QFrame()
        div.setObjectName("vDivider")
        div.setFrameShape(QFrame.Shape.VLine)
        return div

    # ─────────────────────────────────────────────────────────────
    # Source 콤보 관리
    # ─────────────────────────────────────────────────────────────
    def _populate_source_combo(self) -> None:
        self._suppress_combo_event = True
        try:
            self.source_combo.clear()

            # 카메라 자동 탐지
            cams = list_available_cameras(max_index=4)
            for idx in cams:
                self.source_combo.addItem(f"📷  Camera {idx}", idx)

            # 마지막에 사용한 비디오 파일이 있다면 미리 등록
            last_file = app_settings.get_last_video_path()
            if last_file and last_file.exists():
                self.source_combo.addItem(f"📁  {last_file.name}", str(last_file))

            # 항상 마지막 항목으로 "비디오 파일 선택..." 트리거
            self.source_combo.addItem("📁  비디오 파일 선택...", PICK_FILE_SENTINEL)

            # 마지막 사용 카메라가 살아있으면 그걸 선택
            last_cam = app_settings.get_last_camera_index(default=-1)
            for i in range(self.source_combo.count()):
                if self.source_combo.itemData(i) == last_cam:
                    self.source_combo.setCurrentIndex(i)
                    break
        finally:
            self._suppress_combo_event = False

    def _on_source_activated(self, index: int) -> None:
        """사용자가 콤보를 직접 선택했을 때 발화 (programmatic 변경엔 발화 안 함)."""
        if self._suppress_combo_event:
            return
        data = self.source_combo.itemData(index)
        if data != PICK_FILE_SENTINEL:
            return  # 카메라/기존 파일 선택은 별도 처리 불필요

        # ★ 즉시 파일 다이얼로그
        path = self._pick_video_file()
        if path is None:
            # 취소: PICK_FILE 항목이 선택된 채로 두면 어색하므로 첫 항목으로 폴백
            self._suppress_combo_event = True
            try:
                # 카메라가 있으면 첫 카메라, 없으면 마지막에 둔 트리거 항목 직전(기존 파일 항목)
                fallback_idx = 0 if self.source_combo.count() > 1 else 0
                self.source_combo.setCurrentIndex(fallback_idx)
            finally:
                self._suppress_combo_event = False
            return

        # 파일 항목을 콤보에 누적 — 동일 경로가 이미 있으면 그걸로 선택
        existing_idx = self._find_combo_item_by_data(str(path))
        if existing_idx >= 0:
            self.source_combo.setCurrentIndex(existing_idx)
        else:
            # 트리거 항목 바로 위에 삽입 (트리거는 항상 마지막에 유지)
            insert_at = max(0, self.source_combo.count() - 1)
            self._suppress_combo_event = True
            try:
                self.source_combo.insertItem(insert_at, f"📁  {path.name}", str(path))
                self.source_combo.setCurrentIndex(insert_at)
            finally:
                self._suppress_combo_event = False

        app_settings.set_last_video_path(path)

    def _pick_video_file(self) -> Optional[Path]:
        """파일 다이얼로그를 띄우고 선택된 경로 반환 (취소 시 None)."""
        last = app_settings.get_last_video_path()
        start_dir = str(last.parent) if last and last.parent.exists() else str(Path.cwd())
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "비디오 파일 선택",
            start_dir,
            "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        return Path(path_str) if path_str else None

    def _find_combo_item_by_data(self, data) -> int:
        for i in range(self.source_combo.count()):
            if self.source_combo.itemData(i) == data:
                return i
        return -1

    # ─────────────────────────────────────────────────────────────
    # Source 결정 (현재 선택 → FrameSource 인스턴스)
    # ─────────────────────────────────────────────────────────────
    def _resolve_source(self) -> Optional[FrameSource]:
        data = self.source_combo.currentData()
        if data == PICK_FILE_SENTINEL:
            # 트리거 항목이 선택된 채로 Start 가 눌린 비정상 케이스 — 다이얼로그 다시 띄움
            path = self._pick_video_file()
            if path is None:
                return None
            try:
                src = FrameSource(path)
            except RuntimeError as e:
                QMessageBox.warning(self, "오류", str(e))
                return None
            app_settings.set_last_video_path(path)
            return src

        if isinstance(data, str):
            # 비디오 파일 경로
            try:
                return FrameSource(Path(data))
            except RuntimeError as e:
                QMessageBox.warning(self, "오류", str(e))
                return None

        # 카메라 인덱스
        cam_idx = int(data)
        try:
            src = FrameSource(cam_idx)
        except RuntimeError as e:
            QMessageBox.warning(self, "오류", str(e))
            return None
        app_settings.set_last_camera_index(cam_idx)
        return src

    # ─────────────────────────────────────────────────────────────
    # Start / Stop
    # ─────────────────────────────────────────────────────────────
    def _on_start(self) -> None:
        # 1) 엔진 준비
        target_device = self.device_combo.currentData()
        if self._engine is None or str(self._engine.device) != target_device:
            try:
                self._pill_state.set_idle(f"loading {target_device}...")
                QApplication.processEvents()
                self._engine = DesmokeEngine.from_default(device=target_device)
            except Exception as e:
                self._pill_state.set_error("model load failed")
                QMessageBox.critical(self, "모델 로드 실패", f"{e}")
                return

        # 2) 소스 결정
        source = self._resolve_source()
        if source is None:
            self._pill_state.set_idle()
            return

        # 3) 녹화 경로 결정
        record_path: Optional[Path] = None
        if self._record_active:
            record_path = self._ask_record_path()
            if record_path is None:
                source.release()
                self.btn_record.setChecked(False)
                self._record_active = False
                return

        # 4) 비디오 파일 재생일 땐 원본 속도 유지
        target_fps = None
        if isinstance(source.source_repr, str):
            target_fps = source.fps()

        # 5) 워커 시작
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
        self.btn_record.setEnabled(False)
        self._pill_state.set_live(recording=record_path is not None)
        self._pill_frames.setText("Frames  0")
        self._pill_fps.setText("FPS  --")
        self._pill_latency.setText("Latency  --  ms")

    def _on_stop(self) -> None:
        if self._worker is None:
            return
        self.btn_stop.setEnabled(False)
        self._pill_state.set_idle("stopping...")
        self._worker.request_stop()
        self._worker.wait(3000)

    def _ask_record_path(self) -> Optional[Path]:
        default_dir = app_settings.get_last_record_dir(default=Path.cwd())
        suggested = default_dir / f"desmoke_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "녹화 저장 위치", str(suggested), "MP4 (*.mp4)"
        )
        if not path_str:
            return None
        path = Path(path_str)
        app_settings.set_last_record_dir(path.parent)
        return path

    # ─────────────────────────────────────────────────────────────
    # Snapshot / Record toggle
    # ─────────────────────────────────────────────────────────────
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
        if not self._latest_clean_qimg.save(str(path)):
            QMessageBox.warning(self, "오류", f"저장 실패: {path}")

    def _on_record_toggle(self, checked: bool) -> None:
        self._record_active = checked
        self.btn_record.setText("⏺  Recording" if checked else "⏺  Record")

    # ─────────────────────────────────────────────────────────────
    # Worker Slots
    # ─────────────────────────────────────────────────────────────
    def _on_frame(self, orig_qimg, clean_qimg, latency_ms: float) -> None:
        self.preview_orig.update_image(orig_qimg)
        self.preview_clean.update_image(clean_qimg)
        self._latest_clean_qimg = clean_qimg

        # Latency pill 색상 코딩
        self._pill_latency.setText(f"Latency  {latency_ms:.0f}  ms")
        self._pill_latency.setProperty("level", latency_level(latency_ms))
        self._pill_latency.style().unpolish(self._pill_latency)
        self._pill_latency.style().polish(self._pill_latency)

    def _on_stats(self, fps: float, frame_index: int) -> None:
        self._pill_fps.setText(f"FPS  {fps:.1f}")
        self._pill_frames.setText(f"Frames  {frame_index}")

    def _on_worker_finished(self, reason: str) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.btn_record.setEnabled(True)

        if reason == "end_of_stream":
            self._pill_state.set_idle("end of stream")
            QMessageBox.information(self, "완료", "재생이 끝났습니다.")
        elif reason.startswith("error"):
            self._pill_state.set_error("error")
            QMessageBox.warning(self, "처리 중 오류", reason)
        else:
            self._pill_state.set_idle(reason)

    # ─────────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 시그니처)
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(3000)
        super().closeEvent(event)
