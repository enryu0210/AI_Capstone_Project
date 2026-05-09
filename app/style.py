"""
style.py
========
앱 전체에 적용되는 다크 테마 QSS 스타일시트.

[디자인 원칙]
- 의료·수술용 도구에 어울리는 차분한 다크 톤 (피로도 최소화)
- 핵심 액션(Start)에만 강한 강조색을 써서 흐름이 자연스럽게 보이도록
- Record 는 빨간색으로 명확히 구분 (의료기기 표준 색상 관습 따름)
- Latency 등 수치는 색으로 즉시 상태 인지 가능하도록 별도 클래스 부여
"""

from __future__ import annotations


# ── 색상 토큰 (한 곳에서 변경하기 쉽도록 분리) ────────────────────
COLORS = {
    "bg":            "#181826",   # 윈도우 배경
    "surface":       "#22222f",   # 컨트롤 표면
    "surface_alt":   "#2c2c3d",   # 호버/대비
    "surface_dim":   "#0f0f17",   # 프리뷰 검은 배경
    "border":        "#34344a",
    "border_strong": "#4a4a64",
    "text":          "#e6e8ee",
    "text_dim":      "#9ba0b0",
    "text_muted":    "#6b6f80",
    "accent":        "#10b981",   # 의료감 있는 진한 녹청 (Start 버튼)
    "accent_hover":  "#059669",
    "accent_dim":    "#1f3d33",
    "danger":        "#ef4444",   # 녹화중 표시
    "danger_dim":    "#3f1d1d",
    "warn":          "#f59e0b",
    "info":          "#60a5fa",
}


GLOBAL_QSS = f"""
/* ── 전역 ────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: "Segoe UI", "Pretendard", "Malgun Gothic", sans-serif;
    font-size: 10pt;
}}

/* 섹션 라벨 (Source / Device / Actions 머리글) */
QLabel#sectionLabel {{
    color: {COLORS['text_muted']};
    font-size: 8.5pt;
    font-weight: 600;
    letter-spacing: 1px;
    padding-left: 2px;
}}

/* ── 콤보박스 ─────────────────────────────────────────── */
QComboBox {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 22px;
    selection-background-color: {COLORS['accent']};
}}
QComboBox:hover {{ border-color: {COLORS['border_strong']}; }}
QComboBox:focus {{ border-color: {COLORS['accent']}; }}
QComboBox:disabled {{ color: {COLORS['text_muted']}; background-color: {COLORS['surface_dim']}; }}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_dim']};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 6px;
    padding: 4px;
    outline: 0;
}}
QComboBox QAbstractItemView::item {{
    padding: 6px 10px;
    border-radius: 4px;
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {COLORS['accent']};
    color: white;
}}

/* ── 일반 버튼 ────────────────────────────────────────── */
QPushButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: {COLORS['surface_alt']};
    border-color: {COLORS['border_strong']};
}}
QPushButton:pressed {{ background-color: {COLORS['surface_dim']}; }}
QPushButton:disabled {{
    background-color: {COLORS['surface_dim']};
    color: {COLORS['text_muted']};
    border-color: {COLORS['border']};
}}

/* Start: 가장 강조 — 진한 녹청 */
QPushButton#btnStart {{
    background-color: {COLORS['accent']};
    color: white;
    border: 1px solid {COLORS['accent']};
    font-weight: 600;
}}
QPushButton#btnStart:hover {{ background-color: {COLORS['accent_hover']}; }}
QPushButton#btnStart:disabled {{
    background-color: {COLORS['accent_dim']};
    color: {COLORS['text_muted']};
    border-color: {COLORS['accent_dim']};
}}

/* Record (toggle): off=일반, on=빨강 */
QPushButton#btnRecord:checked {{
    background-color: {COLORS['danger']};
    color: white;
    border: 1px solid {COLORS['danger']};
    font-weight: 600;
}}
QPushButton#btnRecord:checked:hover {{ background-color: #dc2626; }}

/* ── 프리뷰 카드 ──────────────────────────────────────── */
QFrame#previewCard {{
    background-color: {COLORS['surface_dim']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}

QLabel#previewTitle {{
    color: {COLORS['text_dim']};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1.2px;
    padding: 10px 14px 8px 14px;
    background-color: transparent;
}}

QLabel#previewBody {{
    background-color: {COLORS['surface_dim']};
    color: {COLORS['text_muted']};
    border: none;
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
}}

/* 헤더 컨트롤바 박스 (Source / Device / Actions 묶음) */
QFrame#headerCard {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}

/* 분리선 */
QFrame#vDivider {{
    background-color: {COLORS['border']};
    max-width: 1px;
    min-width: 1px;
    margin: 8px 4px;
}}

/* ── 상태바 (커스텀 위젯이 들어갈 자리) ───────────────── */
QStatusBar {{
    background-color: {COLORS['surface_dim']};
    border-top: 1px solid {COLORS['border']};
    min-height: 32px;
}}
QStatusBar::item {{ border: none; }}

/* 상태 표시 pill */
QLabel.pill {{
    color: {COLORS['text_dim']};
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 9pt;
}}

QLabel.pillState {{
    color: {COLORS['text']};
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 9pt;
    font-weight: 600;
}}
QLabel.pillState[live="true"] {{
    color: white;
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QLabel.pillState[recording="true"] {{
    color: white;
    background-color: {COLORS['danger']};
    border-color: {COLORS['danger']};
}}
QLabel.pillState[error="true"] {{
    color: {COLORS['danger']};
    background-color: {COLORS['danger_dim']};
    border-color: {COLORS['danger']};
}}

/* Latency 색상 코딩 */
QLabel#pillLatency[level="good"] {{ color: {COLORS['accent']}; }}
QLabel#pillLatency[level="warn"] {{ color: {COLORS['warn']}; }}
QLabel#pillLatency[level="bad"]  {{ color: {COLORS['danger']}; }}

/* ── 메시지 박스 / 다이얼로그 ─────────────────────────── */
QMessageBox {{ background-color: {COLORS['bg']}; }}
QMessageBox QLabel {{ color: {COLORS['text']}; }}
QFileDialog {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
"""


def latency_level(ms: float) -> str:
    """ms 단위 지연을 good/warn/bad 단계로 분류 (UI 색상 결정용)."""
    if ms < 70:
        return "good"
    if ms < 150:
        return "warn"
    return "bad"
