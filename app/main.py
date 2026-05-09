"""
main.py
=======
PFAN 디스모킹 PC 클라이언트 진입점.

실행::

    python -m app.main
"""

from __future__ import annotations

import sys
from pathlib import Path


def _check_checkpoint() -> tuple[bool, str]:
    """체크포인트가 표준 경로에 존재하는지 확인.

    Returns
    -------
    (ok, message): ``ok`` 가 False 면 ``message`` 를 사용자에게 보여주고 종료.
    """
    project_root = Path(__file__).resolve().parent.parent
    expected = (
        project_root
        / "src"
        / "PFAN-SurgiATM-PGSA"
        / "Checkpoint"
        / "PFAN_Final"
        / "270_net_G.pth"
    )
    if expected.exists():
        return True, ""
    msg = (
        "체크포인트가 준비돼 있지 않습니다.\n\n"
        f"기대 경로: {expected}\n\n"
        "다음 명령으로 체크포인트를 표준 경로에 복사한 뒤 다시 실행하세요:\n"
        "    python scripts/prepare_checkpoint.py"
    )
    return False, msg


def main() -> int:
    # ─ 1) 체크포인트 검증 ──────────────────────────────────────
    ok, msg = _check_checkpoint()
    if not ok:
        # GUI 띄우기 전 단계라 PySide 도 import 안 해도 됨 → 빠른 실패.
        print(msg, file=sys.stderr)
        return 1

    # ─ 2) Qt 앱 + 메인 윈도우 ─────────────────────────────────
    # PySide6 import 는 여기에서 (모듈 로드 비용을 늦춤)
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
