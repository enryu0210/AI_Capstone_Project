"""
style.py
========
앱 전체에 적용되는 다크 테마 QSS — "Surgical Telemetry" 에디토리얼 콘솔 룩.

[디자인 톤]
- 의료/수술 콘솔의 계측기 분위기 + 에디토리얼 타이포그래피
- 베이스: 차가운 그래파이트 (#0c0f14) — 일반적인 다크 블루를 피해 정체성 부여
- 활성 강조: 정제된 emerald (#34d399). 녹화는 rose (#f43f5e)
- 디스플레이 폰트: Bahnschrift (Windows 네이티브 condensed sans) — 두께 변주가 풍부해 라벨/제목용
- 데이터 폰트: Consolas — 텔레메트리 숫자에 모노스페이스로 "계측기" 인상

[QSS 제약 메모]
- Qt QSS 는 box-shadow / ::before / ::after 미지원
  → 카드 그림자는 ``QGraphicsDropShadowEffect`` 로 main_window 에서 부여
  → 코너 데코는 ``ViewfinderOverlay`` paintEvent 로 직접 그림
"""

from __future__ import annotations


# ── 색상 토큰 ─────────────────────────────────────────────────────
# 한 곳에서 변경하기 쉽도록 분리. 색은 모두 16진수, 6자리.
COLORS = {
    # 베이스 (cool graphite)
    "bg":             "#0c0f14",
    "bg_inset":       "#080a0e",
    "surface":        "#151a22",
    "surface_alt":    "#1d242e",
    "surface_hover":  "#252d38",
    "surface_dim":    "#06070a",   # 영상 뷰포트용 near-black

    # 보더 / 룰러
    "border":         "#232a34",
    "border_strong":  "#3a4250",
    "hairline":       "#2a3340",   # 상단 하이라이트 (단면 음영용)

    # 텍스트
    "text":           "#e8ebf0",
    "text_dim":       "#8892a0",
    "text_muted":     "#5a6273",
    "text_subtle":    "#3a4250",

    # 강조 (emerald) — 활성 상태와 Start 버튼
    "accent":         "#34d399",
    "accent_hover":   "#10b981",
    "accent_press":   "#059669",
    "accent_dim":     "#1a3a30",
    "accent_ghost":   "#0e2620",

    # 보조 강조 (warm amber) — 호버 포커스 링, 미세한 하이라이트
    "warm":           "#f5d042",
    "warm_dim":       "#3a2e10",

    # 상태
    "danger":         "#f43f5e",
    "danger_hover":   "#e11d48",
    "danger_dim":     "#3a1620",
    "warn":           "#fb923c",
    "info":           "#60a5fa",
}


# ── 폰트 패밀리 (한 곳에서 변경) ───────────────────────────────────
FONTS = {
    # 본문/UI 기본 — Segoe UI 가 가장 안정적
    "body":     '"Segoe UI", "Pretendard", "Malgun Gothic", sans-serif',
    # 디스플레이/섹션 라벨 — Bahnschrift 는 Windows 10+ 네이티브.
    # weight axis 가 있어 SemiLight/SemiBold 가 가능하지만 QSS 에서는
    # font-family 한 줄로만 지정 가능 → Bahnschrift 단일 사용 후 weight 만 조절.
    "display":  '"Bahnschrift Condensed", "Bahnschrift", "Segoe UI Semibold", sans-serif',
    # 텔레메트리 수치 — Consolas 는 Windows 기본. cv1, cv2 같은 변형 옵션은 없지만 충분.
    "mono":     '"Consolas", "Cascadia Mono", "Courier New", monospace',
}


GLOBAL_QSS = f"""
/* ── 전역 ───────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: {FONTS['body']};
    font-size: 10pt;
}}

/* 상단 브랜드 바 — 가로 그라데이션으로 미세한 입체감 */
QFrame#brandBar {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0    {COLORS['surface']},
        stop:0.5  {COLORS['surface_alt']},
        stop:1    {COLORS['surface']}
    );
    border: none;
    border-bottom: 1px solid {COLORS['border']};
}}

QLabel#brandMark {{
    color: {COLORS['accent']};
    font-family: {FONTS['display']};
    font-size: 15pt;
    font-weight: 600;
    letter-spacing: 3px;
    padding-left: 6px;
}}
QLabel#brandTagline {{
    color: {COLORS['text_dim']};
    font-family: {FONTS['display']};
    font-size: 9pt;
    font-weight: 300;
    letter-spacing: 4px;
}}
QLabel#brandVersion {{
    color: {COLORS['text_muted']};
    font-family: {FONTS['mono']};
    font-size: 8.5pt;
    letter-spacing: 1px;
    padding-right: 4px;
}}

/* 섹션 라벨 (INPUT / DEVICE / ACTIONS) — 작은 캡스 분위기 */
QLabel#sectionLabel {{
    color: {COLORS['text_muted']};
    font-family: {FONTS['display']};
    font-size: 8.5pt;
    font-weight: 600;
    letter-spacing: 3px;
    padding-left: 2px;
}}

/* ── 콤보박스 ───────────────────────────────────────────────── */
QComboBox {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 7px 12px;
    min-height: 24px;
    selection-background-color: {COLORS['accent']};
}}
QComboBox:hover {{
    border-color: {COLORS['border_strong']};
    background-color: {COLORS['surface_hover']};
}}
QComboBox:focus {{ border-color: {COLORS['accent']}; }}
QComboBox:disabled {{
    color: {COLORS['text_muted']};
    background-color: {COLORS['surface_dim']};
    border-color: {COLORS['border']};
}}
QComboBox::drop-down {{
    border: none;
    width: 26px;
    subcontrol-origin: padding;
    subcontrol-position: top right;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_dim']};
    margin-right: 10px;
}}
QComboBox::down-arrow:hover {{
    border-top-color: {COLORS['accent']};
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 4px;
    padding: 4px;
    outline: 0;
}}
QComboBox QAbstractItemView::item {{
    padding: 7px 10px;
    border-radius: 3px;
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {COLORS['accent_dim']};
    color: {COLORS['accent']};
}}

/* ── 일반 버튼 (Stop / Snapshot / Record off) ──────────────── */
QPushButton {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px 14px;
    font-weight: 500;
    letter-spacing: 0.5px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {COLORS['surface_hover']};
    border-color: {COLORS['border_strong']};
    color: {COLORS['warm']};
}}
QPushButton:pressed {{
    background-color: {COLORS['surface_dim']};
}}
QPushButton:disabled {{
    background-color: {COLORS['surface_dim']};
    color: {COLORS['text_subtle']};
    border-color: {COLORS['border']};
}}

/* Start 버튼 — 가장 강한 강조. emerald fill + 미세 글로우 효과는 QSS 한계로
   border 와 색상 조합으로 표현 */
QPushButton#btnStart {{
    background-color: {COLORS['accent']};
    color: #06231b;
    border: 1px solid {COLORS['accent']};
    font-weight: 700;
    letter-spacing: 1.2px;
}}
QPushButton#btnStart:hover {{
    background-color: {COLORS['accent_hover']};
    color: #ffffff;
    border-color: {COLORS['accent_hover']};
}}
QPushButton#btnStart:pressed {{
    background-color: {COLORS['accent_press']};
}}
QPushButton#btnStart:disabled {{
    background-color: {COLORS['accent_ghost']};
    color: {COLORS['text_muted']};
    border-color: {COLORS['accent_dim']};
}}

/* Record 토글 — off 일 때는 일반 버튼처럼, on 일 때 강한 빨강 */
QPushButton#btnRecord:checked {{
    background-color: {COLORS['danger']};
    color: #ffffff;
    border: 1px solid {COLORS['danger']};
    font-weight: 700;
    letter-spacing: 1.2px;
}}
QPushButton#btnRecord:checked:hover {{
    background-color: {COLORS['danger_hover']};
}}

/* Stop 버튼 — 비활성 시 더욱 약하게 */
QPushButton#btnStop:disabled {{
    color: {COLORS['text_subtle']};
}}

/* ── 컨트롤 카드 (헤더) ───────────────────────────────────── */
QFrame#headerCard {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0  {COLORS['surface_alt']},
        stop:1  {COLORS['surface']}
    );
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}

/* 수직 분리선 — 약간의 그라데이션을 줘서 단순한 선보다 깊이감 */
QFrame#vDivider {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0    transparent,
        stop:0.5  {COLORS['border_strong']},
        stop:1    transparent
    );
    max-width: 1px;
    min-width: 1px;
    margin: 6px 6px;
    border: none;
}}

/* ── 프리뷰 카드 ───────────────────────────────────────────── */
QFrame#previewCard {{
    background-color: {COLORS['surface_dim']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}

/* 카드 헤더 스트립 — 좌측에 액센트 컬러 인디케이터 라인 */
QFrame#previewHeader {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0    {COLORS['surface']},
        stop:1    {COLORS['surface_alt']}
    );
    border: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}

QLabel#previewTitleMark {{
    color: {COLORS['accent']};
    font-family: {FONTS['display']};
    font-size: 11pt;
    font-weight: 600;
    padding-right: 4px;
}}
QLabel#previewTitleMark[role="muted"] {{
    color: {COLORS['text_muted']};
}}

QLabel#previewTitle {{
    color: {COLORS['text']};
    font-family: {FONTS['display']};
    font-size: 9.5pt;
    font-weight: 600;
    letter-spacing: 3px;
}}

QLabel#previewTag {{
    color: {COLORS['text_muted']};
    font-family: {FONTS['mono']};
    font-size: 8pt;
    letter-spacing: 1px;
    padding-right: 6px;
}}

QLabel#previewBody {{
    background-color: {COLORS['surface_dim']};
    color: {COLORS['text_muted']};
    font-family: {FONTS['display']};
    font-size: 10pt;
    letter-spacing: 1px;
    border: none;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}}

/* ── 상태바 ────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {COLORS['bg_inset']};
    border-top: 1px solid {COLORS['border']};
    min-height: 36px;
    padding: 0 6px;
}}
QStatusBar::item {{ border: none; }}

/* 상태바 내부 분리선 (텔레메트리 그룹 사이) */
QFrame#statusDivider {{
    background-color: {COLORS['border']};
    max-width: 1px;
    min-width: 1px;
    margin: 8px 4px;
    border: none;
}}

/* 텔레메트리 그룹 — 라벨 + 모노 숫자 두 줄 구조 */
QLabel#telemetryLabel {{
    color: {COLORS['text_muted']};
    font-family: {FONTS['display']};
    font-size: 7.5pt;
    font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#telemetryValue {{
    color: {COLORS['text']};
    font-family: {FONTS['mono']};
    font-size: 11pt;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QLabel#telemetryValue[level="good"] {{ color: {COLORS['accent']}; }}
QLabel#telemetryValue[level="warn"] {{ color: {COLORS['warn']}; }}
QLabel#telemetryValue[level="bad"]  {{ color: {COLORS['danger']}; }}

/* 상태 표시 pill — 좌측 라이브 인디케이터 */
QLabel#pillState {{
    color: {COLORS['text_dim']};
    background-color: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 13px;
    padding: 5px 14px;
    font-family: {FONTS['display']};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#pillState[live="true"] {{
    color: #06231b;
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QLabel#pillState[recording="true"] {{
    color: #ffffff;
    background-color: {COLORS['danger']};
    border-color: {COLORS['danger']};
}}
QLabel#pillState[error="true"] {{
    color: {COLORS['danger']};
    background-color: {COLORS['danger_dim']};
    border-color: {COLORS['danger']};
}}

/* ── 체크박스 (Detector 토글) ──────────────────────────────── */
QCheckBox#chkDetector {{
    color: {COLORS['text']};
    font-family: {FONTS['display']};
    font-size: 9pt;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 2px 0;
    spacing: 8px;
}}
QCheckBox#chkDetector:disabled {{
    color: {COLORS['text_subtle']};
}}
QCheckBox#chkDetector::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {COLORS['border_strong']};
    border-radius: 3px;
    background-color: {COLORS['surface_alt']};
}}
QCheckBox#chkDetector::indicator:hover {{
    border-color: {COLORS['accent']};
}}
QCheckBox#chkDetector::indicator:checked {{
    border-color: {COLORS['accent']};
    background-color: {COLORS['accent']};
}}
QCheckBox#chkDetector::indicator:disabled {{
    background-color: {COLORS['surface_dim']};
    border-color: {COLORS['border']};
}}

/* ── Smoke 인디케이터 pill ─────────────────────────────────── */
QLabel#pillSmoke {{
    color: {COLORS['text_dim']};
    background-color: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 13px;
    padding: 5px 14px;
    font-family: {FONTS['display']};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#pillSmoke[state="off"] {{
    color: {COLORS['text_subtle']};
    background-color: {COLORS['surface_dim']};
    border-color: {COLORS['border']};
}}
QLabel#pillSmoke[state="clear"] {{
    color: #06231b;
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QLabel#pillSmoke[state="smoke"] {{
    color: #ffffff;
    background-color: {COLORS['danger']};
    border-color: {COLORS['danger']};
}}

/* ── 탭 위젯 (LIVE / ANALYSIS) ─────────────────────────────── */
/* documentMode=True 이므로 탭바만 보이고 패널 보더는 직접 그리지 않음. */
QTabWidget#previewTabs::pane {{
    border: none;
    top: -1px;            /* 탭과 컨텐츠 사이의 1px 틈 제거 */
}}
QTabWidget#previewTabs QTabBar {{
    qproperty-drawBase: 0;
}}
QTabWidget#previewTabs QTabBar::tab {{
    background-color: transparent;
    color: {COLORS['text_muted']};
    font-family: {FONTS['display']};
    font-size: 9.5pt;
    font-weight: 600;
    letter-spacing: 4px;
    padding: 8px 22px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-bottom: 1px solid {COLORS['border']};
}}
QTabWidget#previewTabs QTabBar::tab:hover {{
    color: {COLORS['text']};
}}
QTabWidget#previewTabs QTabBar::tab:selected {{
    color: {COLORS['accent']};
    /* 선택된 탭의 하단에만 accent 색 인디케이터 라인 */
    border-bottom: 2px solid {COLORS['accent']};
}}

/* ── 다이얼로그 / 메시지 박스 ─────────────────────────────── */
QMessageBox {{ background-color: {COLORS['bg']}; }}
QMessageBox QLabel {{ color: {COLORS['text']}; font-family: {FONTS['body']}; }}
QFileDialog {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}

/* 툴팁 — 거의 모든 곳에 적용 */
QToolTip {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 4px;
    padding: 6px 10px;
    font-family: {FONTS['body']};
}}
"""


def latency_level(ms: float) -> str:
    """ms 단위 지연을 good/warn/bad 단계로 분류 (UI 색상 결정용)."""
    if ms < 70:
        return "good"
    if ms < 150:
        return "warn"
    return "bad"
