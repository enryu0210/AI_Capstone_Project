"""
prepare_checkpoint.py
=====================
PFAN 디스모킹 PC 클라이언트가 사용할 체크포인트를 표준 경로로 복사하는 1회용 부트스트랩.

[왜 필요한가]
- 학습 코드(`src/PFAN-SurgiATM-PGSA/models/base_model.py:load_networks`)는
  체크포인트를 항상 `<checkpoints_dir>/<name>/<epoch>_net_G.pth` 형식으로 찾습니다.
- 그런데 실제 가중치 파일은 `Checkpoint/Final 270_net_G.pth` 처럼 표준 경로 밖에
  보관돼 있어, 그대로는 표준 로더가 인식하지 못합니다.
- 매 실행마다 로더를 패치하기보다, 한 번만 표준 경로(`Checkpoint/PFAN_Final/270_net_G.pth`)
  로 복사해두면 이후 추론 코드가 깔끔해집니다.

[실행]
    python scripts/prepare_checkpoint.py

[안전성]
- 원본 파일은 그대로 보존 (shutil.copy2 = 복사이며 메타데이터까지 보존)
- 대상 파일이 이미 있으면 건너뜀 (idempotent)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ── 경로 정의 ────────────────────────────────────────────────────────
# 이 스크립트는 프로젝트 루트(scripts/ 의 부모) 기준으로 동작하도록 설계.
# 실행 디렉토리가 어디든 안전하게 절대 경로로 변환한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 원본: PFAN 학습 결과물의 최종 가중치 (사용자가 사전에 배치한 파일)
SRC_CHECKPOINT = (
    PROJECT_ROOT
    / "src"
    / "PFAN-SurgiATM-PGSA"
    / "Checkpoint"
    / "Final 270_net_G.pth"
)

# 대상: 표준 로더가 자동으로 찾는 경로
#   - 'PFAN_Final' = 학습 시 --name 으로 줄 이름과 동일한 의미
#   - '270_net_G.pth' = epoch=270 의 Generator 가중치
DST_DIR = (
    PROJECT_ROOT
    / "src"
    / "PFAN-SurgiATM-PGSA"
    / "Checkpoint"
    / "PFAN_Final"
)
DST_CHECKPOINT = DST_DIR / "270_net_G.pth"


def main() -> int:
    """체크포인트 복사. 성공: 0, 실패: 1."""
    print(f"[prepare_checkpoint] 프로젝트 루트: {PROJECT_ROOT}")

    # 1) 원본 존재 확인 — 가장 흔한 실수(파일 이름 오타 등) 방지
    if not SRC_CHECKPOINT.exists():
        print(
            f"[ERROR] 원본 체크포인트를 찾을 수 없습니다:\n"
            f"        {SRC_CHECKPOINT}\n"
            f"        Final 270_net_G.pth 파일이 위 경로에 있는지 확인하세요.",
            file=sys.stderr,
        )
        return 1

    # 2) 이미 복사돼 있다면 스킵 (Idempotent)
    if DST_CHECKPOINT.exists():
        # 크기까지 확인하면 더 안전하지만, 동일 파일이라 가정 (사용자가 임의로
        # 손대지 않았다는 전제). 의심스러우면 사용자가 수동 삭제하면 됨.
        print(f"[OK] 이미 복사돼 있습니다: {DST_CHECKPOINT}")
        return 0

    # 3) 디렉토리 생성 + 파일 복사
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 복사 중...\n  src: {SRC_CHECKPOINT}\n  dst: {DST_CHECKPOINT}")
    try:
        shutil.copy2(SRC_CHECKPOINT, DST_CHECKPOINT)
    except OSError as e:
        # 디스크 가득참, 권한 부족 등
        print(f"[ERROR] 복사 실패: {e}", file=sys.stderr)
        return 1

    # 4) 사후 검증 — 복사가 정상적으로 끝났는지 크기 비교
    if DST_CHECKPOINT.stat().st_size != SRC_CHECKPOINT.stat().st_size:
        print(
            "[ERROR] 복사된 파일 크기가 원본과 다릅니다. 다시 시도해 주세요.",
            file=sys.stderr,
        )
        return 1

    print("[DONE] 체크포인트 준비 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
