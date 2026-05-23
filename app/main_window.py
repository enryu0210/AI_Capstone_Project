"""
main_window.py
==============
PFAN 디스모킹 PC 클라이언트의 메인 윈도우 — "Surgical Telemetry" 디자인.

[레이아웃 개요]
    ┌─ Brand bar ───────────────────────────────────────────────┐
    │  PFAN ·  DESMOKE      SURGICAL VIDEO PROCESSING    v1.0    │
    ├────────────────────────────────────────────────────────────┤
    │ ┌─ Header card ────────────────────────────────────────┐   │
    │ │ INPUT | DEVICE | ACTIONS                             │   │
    │ │ [📷Camera▾]  [⚡GPU▾]   [▶ Start][⏹ Stop][⏺Rec][📷]   │   │
    │ └──────────────────────────────────────────────────────┘   │
    │                                                            │
    │ ┌─ ◉ ORIGINAL ─────────┐  ┌─ ✓ DESMOKED ────────┐         │
    │ │ ⌐               ¬    │  │ ⌐               ¬    │         │
    │ │   (영상 표시 영역)    │  │   (영상 표시 영역)    │         │
    │ │ ⌊               ⌋    │  │ ⌊               ⌋    │         │
    │ └──────────────────────┘  └──────────────────────┘         │
    │                                                            │
    │ ● LIVE   │ FPS  14.2 │ LAT  67 ms │ FRM 1247                │
    └────────────────────────────────────────────────────────────┘

[디자인 메모]
- 카드 그림자는 ``QGraphicsDropShadowEffect`` 로 — QSS 가 box-shadow 미지원
- 프리뷰 카드의 코너 데코는 ``ViewfinderOverlay`` 가 paintEvent 에서 직접 그림
- 텔레메트리는 라벨 + 모노스페이스 숫자의 2단 구조 (계측기 분위기)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app import settings as app_settings
from app.desmoke_engine import DesmokeEngine
from app.frame_source import FrameSource, list_available_cameras
from app.inference_worker import InferenceWorker
from app.style import COLORS, GLOBAL_QSS, latency_level


# ── 콤보박스 데이터 sentinel ────────────────────────────────────
# - int (>= 0)  : 카메라 인덱스
# - "PICK_FILE" : 파일 다이얼로그를 띄우는 트리거 항목
# - 문자열 path : 사용자가 선택한 비디오 파일 경로 (실제 소스)
PICK_FILE_SENTINEL = "PICK_FILE"


# ──────────────────────────────────────────────────────────────────
# 유틸: 카드용 드롭 섀도우 효과
# ──────────────────────────────────────────────────────────────────
def _attach_card_shadow(widget: QWidget, *, strong: bool = False) -> None:
    """카드 뒤에 부드러운 그림자를 깔아 깊이감을 부여한다.

    QSS 는 box-shadow 를 지원하지 않으므로 그래픽스 이펙트로 구현.
    """
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(28 if strong else 18)
    eff.setOffset(0, 6 if strong else 4)
    eff.setColor(QColor(0, 0, 0, 160 if strong else 110))
    widget.setGraphicsEffect(eff)


# ──────────────────────────────────────────────────────────────────
# 프리뷰: 카메라 뷰파인더 같은 코너 틱 오버레이
# ──────────────────────────────────────────────────────────────────
class ViewfinderOverlay(QWidget):
    """프리뷰 영역 위에 떠 있는 투명 오버레이 — 네 모서리에 L 자형 마커를 그림.

    Why: 단순 사각 프레임보다 "계측기/카메라" 인상이 강해져 의료 장비 톤과
         잘 어울린다. 이미지 자체는 가리지 않도록 마우스 이벤트는 통과시킴.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._color = QColor(COLORS["accent"])
        self._color.setAlpha(170)  # 너무 강하면 영상에 방해되므로 살짝 투명

    def set_active(self, active: bool) -> None:
        """LIVE 시 강조색, idle 시 흐린 회색."""
        if active:
            self._color = QColor(COLORS["accent"])
            self._color.setAlpha(190)
        else:
            self._color = QColor(COLORS["text_subtle"])
            self._color.setAlpha(150)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt 시그니처)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self._color)
        pen.setWidthF(1.6)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)

        # 코너 안쪽으로 약간 여백을 두고, 각 모서리에 L 자 두 선을 그림
        m = 10.0      # 안쪽 마진
        L = 22.0      # L 자 한 변 길이
        w = float(self.width())
        h = float(self.height())

        # 좌상
        painter.drawLine(QRectF(m, m, L, 0).topLeft(), QRectF(m, m, L, 0).topRight())
        painter.drawLine(QRectF(m, m, 0, L).topLeft(), QRectF(m, m, 0, L).bottomLeft())
        # 우상
        painter.drawLine(QRectF(w - m - L, m, L, 0).topLeft(), QRectF(w - m - L, m, L, 0).topRight())
        painter.drawLine(QRectF(w - m, m, 0, L).topLeft(), QRectF(w - m, m, 0, L).bottomLeft())
        # 좌하
        painter.drawLine(QRectF(m, h - m - L, 0, L).topLeft(), QRectF(m, h - m - L, 0, L).bottomLeft())
        painter.drawLine(QRectF(m, h - m, L, 0).topLeft(), QRectF(m, h - m, L, 0).topRight())
        # 우하
        painter.drawLine(QRectF(w - m, h - m - L, 0, L).topLeft(), QRectF(w - m, h - m, 0, L).bottomLeft())
        painter.drawLine(QRectF(w - m - L, h - m, L, 0).topLeft(), QRectF(w - m - L, h - m, L, 0).topRight())


# ──────────────────────────────────────────────────────────────────
# 프리뷰 카드 (헤더 스트립 + 영상 영역 + 코너 오버레이)
# ──────────────────────────────────────────────────────────────────
class PreviewCard(QFrame):
    """제목 스트립 + 영상 표시 영역으로 구성된 카드형 프리뷰 위젯."""

    def __init__(self, mark: str, title: str, tag: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("previewCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        # 영상이 카드 모서리 라운드 밖으로 비져 나오지 않게
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 헤더 스트립 ─────────────────────────────────────
        header = QFrame()
        header.setObjectName("previewHeader")
        header.setFixedHeight(38)
        h = QHBoxLayout(header)
        h.setContentsMargins(14, 0, 12, 0)
        h.setSpacing(6)

        self._mark = QLabel(mark)
        self._mark.setObjectName("previewTitleMark")
        self._title = QLabel(title.upper())
        self._title.setObjectName("previewTitle")
        self._tag = QLabel(tag.upper())
        self._tag.setObjectName("previewTag")

        h.addWidget(self._mark)
        h.addWidget(self._title)
        h.addStretch(1)
        h.addWidget(self._tag)

        layout.addWidget(header)

        # ── 영상 표시 영역 (body) + 오버레이 ─────────────────
        # body 와 overlay 를 같은 영역에 겹치기 위해 QFrame 컨테이너 사용.
        stage = QFrame()
        stage.setObjectName("previewStage")
        stage.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        stage.setStyleSheet(
            f"QFrame#previewStage {{ background-color: {COLORS['surface_dim']}; "
            f"border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }}"
        )

        self._body = QLabel(subtitle, stage)
        self._body.setObjectName("previewBody")
        self._body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self._overlay = ViewfinderOverlay(stage)

        # stage 가 리사이즈될 때 body 와 overlay 모두 채워지도록
        stage.installEventFilter(self)
        self._stage = stage
        stage.setMinimumSize(360, 240)

        layout.addWidget(stage, 1)

    # ── stage 리사이즈에 맞춰 body / overlay 크기 동기화 ─────
    def eventFilter(self, obj, event):  # noqa: N802
        from PySide6.QtCore import QEvent
        if obj is self._stage and event.type() == QEvent.Type.Resize:
            r = self._stage.rect()
            self._body.setGeometry(r)
            self._overlay.setGeometry(r)
            self._overlay.raise_()
        return super().eventFilter(obj, event)

    def sizeHint(self) -> QSize:
        return QSize(640, 380)

    def set_active(self, active: bool) -> None:
        """LIVE 상태에 따라 헤더 마커와 코너 틱 색을 바꿈."""
        self._mark.setProperty("role", "" if active else "muted")
        self._mark.style().unpolish(self._mark)
        self._mark.style().polish(self._mark)
        self._overlay.set_active(active)

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


# ──────────────────────────────────────────────────────────────────
# 상태 / 텔레메트리 위젯
# ──────────────────────────────────────────────────────────────────
class StatePill(QLabel):
    """상태바 좌측의 상태 표시 — running/recording/error 등 동적으로 색이 바뀜."""

    def __init__(self, parent=None) -> None:
        super().__init__("● IDLE", parent)
        self.setObjectName("pillState")

    def set_idle(self, text: str = "idle") -> None:
        self._reset_flags()
        self.setText(f"● {text.upper()}")
        self._refresh_style()

    def set_live(self, recording: bool) -> None:
        self._reset_flags()
        if recording:
            self.setProperty("recording", "true")
            self.setText("⏺ REC · LIVE")
        else:
            self.setProperty("live", "true")
            self.setText("● LIVE")
        self._refresh_style()

    def set_error(self, text: str) -> None:
        self._reset_flags()
        self.setProperty("error", "true")
        self.setText(f"⚠ {text.upper()}")
        self._refresh_style()

    def _reset_flags(self) -> None:
        for k in ("live", "recording", "error"):
            self.setProperty(k, "")

    def _refresh_style(self) -> None:
        # QSS property 변경 후 강제 재적용
        self.style().unpolish(self)
        self.style().polish(self)


class TelemetryBlock(QWidget):
    """라벨(작은 캡스) + 값(모노스페이스) 2단 구조의 텔레메트리 표시.

    예) FPS / 14.2  — 라벨은 톤다운, 값은 큰 모노 폰트로 계측기 분위기.
    """

    def __init__(self, label: str, value: str = "—", object_name: str = "", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        self._label = QLabel(label.upper())
        self._label.setObjectName("telemetryLabel")
        self._value = QLabel(value)
        self._value.setObjectName("telemetryValue")
        if object_name:
            self._value.setObjectName(object_name)
            # objectName 으로도 식별 가능하도록 추가 클래스를 부여
            self._value.setProperty("class", "telemetryValue")

        layout.addWidget(self._label)
        layout.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)

    def set_level(self, level: str) -> None:
        """latency 단계 (good/warn/bad) 에 따른 색 갱신."""
        self._value.setProperty("level", level)
        self._value.style().unpolish(self._value)
        self._value.style().polish(self._value)


# ──────────────────────────────────────────────────────────────────
# 메인 윈도우
# ──────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PFAN · DESMOKE — Surgical Video Processing")
        self.resize(1360, 820)
        self.setStyleSheet(GLOBAL_QSS)

        # ── 상태 ────────────────────────────────────────────────
        self._engine: Optional[DesmokeEngine] = None
        self._worker: Optional[InferenceWorker] = None
        self._latest_clean_qimg: Optional[QImage] = None
        self._record_active: bool = False
        # 콤보 selection 변경을 사용자 액션과 프로그래매틱 변경을 구분
        self._suppress_combo_event: bool = False

        # ── 중앙 위젯 / 루트 레이아웃 ──────────────────────────
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 1) 브랜드 바
        root.addWidget(self._build_brand_bar())

        # 2) 컨텐츠 영역 (헤더 카드 + 프리뷰)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(18, 16, 18, 12)
        content_layout.setSpacing(14)

        header_card = self._build_header_card()
        _attach_card_shadow(header_card)
        content_layout.addWidget(header_card)

        content_layout.addLayout(self._build_previews(), 1)
        root.addWidget(content, 1)

        # 3) 상태바
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
    def _build_brand_bar(self) -> QFrame:
        """최상단 브랜드 바 — PFAN · DESMOKE 워드마크 + 태그라인."""
        bar = QFrame()
        bar.setObjectName("brandBar")
        bar.setFixedHeight(54)

        row = QHBoxLayout(bar)
        row.setContentsMargins(22, 0, 22, 0)
        row.setSpacing(0)

        mark = QLabel("PFAN")
        mark.setObjectName("brandMark")

        sep = QLabel("·")
        sep.setObjectName("brandSeparator")

        product = QLabel("DESMOKE")
        product.setObjectName("brandMark")
        # 워드마크 두 단어 사이 시각적 균형을 위해 더 톤다운된 색이 자연스러움
        product.setStyleSheet(f"color: {COLORS['text']}; letter-spacing: 4px;")

        tagline = QLabel("SURGICAL  VIDEO  PROCESSING")
        tagline.setObjectName("brandTagline")

        version = QLabel("v1.0")
        version.setObjectName("brandVersion")

        row.addWidget(mark)
        row.addWidget(sep)
        row.addWidget(product)
        row.addSpacing(28)
        # 얇은 세로 구분선 — 브랜드와 태그라인 사이
        sep_line = QFrame()
        sep_line.setObjectName("vDivider")
        sep_line.setFixedSize(1, 22)
        sep_line.setStyleSheet(f"background-color: {COLORS['border_strong']};")
        row.addWidget(sep_line)
        row.addSpacing(16)
        row.addWidget(tagline)
        row.addStretch(1)
        row.addWidget(version)

        return bar

    def _build_header_card(self) -> QFrame:
        """상단 컨트롤 카드 (Source / Device / Actions)."""
        card = QFrame()
        card.setObjectName("headerCard")

        outer = QHBoxLayout(card)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(14)

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
        wrap.setSpacing(6)
        title = QLabel("INPUT")
        title.setObjectName("sectionLabel")
        wrap.addWidget(title)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(280)
        self.source_combo.setToolTip("카메라 / 비디오 파일 선택")
        wrap.addWidget(self.source_combo)
        return wrap

    def _build_device_group(self) -> QVBoxLayout:
        wrap = QVBoxLayout()
        wrap.setSpacing(6)
        title = QLabel("DEVICE")
        title.setObjectName("sectionLabel")
        wrap.addWidget(title)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(180)
        self.device_combo.addItem("⚡  GPU (cuda:0)", "cuda:0")
        self.device_combo.addItem("🖥  CPU", "cpu")
        self.device_combo.setToolTip("추론 디바이스 선택")
        wrap.addWidget(self.device_combo)
        return wrap

    def _build_actions_group(self) -> QVBoxLayout:
        wrap = QVBoxLayout()
        wrap.setSpacing(6)
        title = QLabel("ACTIONS")
        title.setObjectName("sectionLabel")
        wrap.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start = QPushButton("▶  START")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setToolTip("디스모킹 추론 시작")
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setToolTip("추론 중단")
        self.btn_record = QPushButton("⏺  REC")
        self.btn_record.setObjectName("btnRecord")
        self.btn_record.setCheckable(True)
        self.btn_record.setToolTip("결과 영상 mp4 로 녹화")
        self.btn_snapshot = QPushButton("📷  SNAP")
        self.btn_snapshot.setObjectName("btnSnapshot")
        self.btn_snapshot.setToolTip("현재 디스모킹 결과 프레임을 PNG/JPG 로 저장")

        for btn in (self.btn_start, self.btn_stop, self.btn_record, self.btn_snapshot):
            btn.setMinimumWidth(108)
            row.addWidget(btn)

        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

        wrap.addLayout(row)
        return wrap

    def _build_previews(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(14)

        self.preview_orig = PreviewCard(
            mark="◉",
            title="Original",
            tag="SOURCE",
            subtitle="SOURCE 를 선택한 뒤  ▶ START  를 누르세요",
        )
        self.preview_clean = PreviewCard(
            mark="✓",
            title="Desmoked",
            tag="PFAN · SURGIATM",
            subtitle="추론 결과가 여기에 표시됩니다",
        )
        # 시작 전엔 비활성 톤
        self.preview_orig.set_active(False)
        self.preview_clean.set_active(False)

        # 카드에 그림자 — 떠 있는 인상
        _attach_card_shadow(self.preview_orig, strong=True)
        _attach_card_shadow(self.preview_clean, strong=True)

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
        bar.addWidget(self._make_status_divider())

        # 우측 (영구 위젯): 텔레메트리 블록들 (오른쪽부터 역순으로 쌓임)
        self._tele_frames = TelemetryBlock("Frames", "0")
        self._tele_latency = TelemetryBlock("Latency · ms", "—")
        self._tele_fps = TelemetryBlock("FPS", "—")

        bar.addPermanentWidget(self._tele_fps)
        bar.addPermanentWidget(self._make_status_divider())
        bar.addPermanentWidget(self._tele_latency)
        bar.addPermanentWidget(self._make_status_divider())
        bar.addPermanentWidget(self._tele_frames)

        return bar

    def _make_divider(self) -> QFrame:
        div = QFrame()
        div.setObjectName("vDivider")
        # headerCard 안에서 충분히 보이도록 살짝 길게
        div.setMinimumHeight(48)
        return div

    def _make_status_divider(self) -> QFrame:
        div = QFrame()
        div.setObjectName("statusDivider")
        div.setMinimumHeight(28)
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
            # 취소 시: PICK_FILE 항목이 선택된 채로 두면 어색하므로 첫 항목으로 폴백
            self._suppress_combo_event = True
            try:
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
        self.preview_orig.set_active(True)
        self.preview_clean.set_active(True)
        self._tele_frames.set_value("0")
        self._tele_fps.set_value("—")
        self._tele_latency.set_value("—")
        self._tele_latency.set_level("")

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
        self.btn_record.setText("⏺  RECORDING" if checked else "⏺  REC")

    # ─────────────────────────────────────────────────────────────
    # Worker Slots
    # ─────────────────────────────────────────────────────────────
    def _on_frame(self, orig_qimg, clean_qimg, latency_ms: float) -> None:
        self.preview_orig.update_image(orig_qimg)
        self.preview_clean.update_image(clean_qimg)
        self._latest_clean_qimg = clean_qimg

        # Latency 텔레메트리 + 색상 코딩
        self._tele_latency.set_value(f"{latency_ms:.0f}")
        self._tele_latency.set_level(latency_level(latency_ms))

    def _on_stats(self, fps: float, frame_index: int) -> None:
        self._tele_fps.set_value(f"{fps:.1f}")
        self._tele_frames.set_value(f"{frame_index}")

    def _on_worker_finished(self, reason: str) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.preview_orig.set_active(False)
        self.preview_clean.set_active(False)

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
