"""
desmoke_engine.py
=================
PFAN+SurgiATM 모델을 GUI 등 외부에서 손쉽게 호출할 수 있도록 감싼 추론 엔진.

[설계 의도]
- 학습 코드(`Pix2PixModel`)는 argparse 와 폴더 dataset 에 강하게 결합돼 있음.
- 그대로 GUI 에서 쓰면 매 프레임마다 폴더를 만들 수 없으니, 모델을 한번만
  로드해 두고 텐서를 직접 주입해 추론하는 방식으로 우회한다.
- `set_input(dict)` 가 받는 형식만 충족하면 dataset 없이 호출 가능 → 실시간 OK.

[추론 파이프라인 (Pix2PixModel.forward 와 동일)]
    smoky [-1,1] → SurgiATM 으로 D_refined 추출 → PFAN 으로 rho_DNN 예측
    → 물리 합성 (smoky_norm − (η+D_refined)/(η+1) * (1−rho_DNN_norm))
    → clamp([0,1]) → 다시 [-1,1] 로 매핑된 fake_B 반환
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

# ── PFAN 코드베이스 import 경로 등록 ────────────────────────────────
# Pix2PixModel 등이 `from models.SurgiATM import ...` 처럼 절대경로 형태로
# import 하고 있어, 해당 디렉토리를 sys.path 에 직접 추가해야 한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PFAN_ROOT = PROJECT_ROOT / "src" / "PFAN-SurgiATM-PGSA"
if str(PFAN_ROOT) not in sys.path:
    sys.path.insert(0, str(PFAN_ROOT))


# 위 경로를 등록한 뒤에야 안전하게 import 가능 — 순서 주의.
# noqa 주석은 모듈 임포트가 sys.path 조작 후에 와야 하기 때문에 필요.
from models.pix2pix_model import Pix2PixModel  # noqa: E402
from util.util import tensor2im  # noqa: E402


# 학습 시 사용된 옵션값 (Checkpoint/PFAN_InVivo/train_opt.txt 그대로 옮김).
# Pix2PixModel 이 실제로 참조하는 키만 모아 SimpleNamespace 로 구성한다.
# 새 키가 필요할 때마다 여기 추가하면 됨.
_TRAIN_OPT_DEFAULTS = dict(
    # ─ 모델 구조 ────────────────────────────────────────────────────
    model="pix2pix",
    netG="pfan",
    netD="basic",          # test 단계엔 D 가 안 만들어지지만, 옵션값은 일관되게 유지
    norm="batch",
    # ※ train_opt.txt 에는 input_nc=3 으로 기록돼 있으나, 실제 학습된 가중치
    #   `model_1.0.weight` 는 (64, 4, 1, 1) 로 저장돼 있음. PFAN.forward 가
    #   RGB(3) + D_refined(1) = 4채널을 cat 한 뒤 첫 conv 에 통과시키기 때문이며,
    #   이 conv 가 사실상 입력 4채널을 받는다. 따라서 추론 시에는 4 로 지정해야
    #   체크포인트 shape 와 정확히 맞아 RuntimeError 없이 로드된다.
    input_nc=4,
    output_nc=3,
    ngf=64,
    ndf=16,
    n_layers_D=3,
    init_type="kaiming",
    init_gain=0.02,
    no_dropout=False,
    # ─ 트랜스포머 옵션 ─────────────────────────────────────────────
    embed_dim=64,
    win_size=8,
    token_projection="conv",
    dd_in=3,
    use_norm=1,
    syn_norm=False,
    train_ps=256,
    # ─ 데이터셋 / 방향 ─────────────────────────────────────────────
    dataset_mode="aligned",
    direction="AtoB",
    # ─ 손실 (test 시 사용 안 됨이지만 Pix2PixModel.__init__ 가 참조) ──
    lambda_L1=100.0,
    surgiatm_wz=15,
    lambda_smooth=1.2434794417055085,
    smooth_alpha=49.843750632553075,
    gan_mode="lsgan",
    # ─ 기타 (Pix2PixModel / BaseModel 이 참조) ───────────────────────
    isTrain=False,
    verbose=False,
    suffix="",
    load_iter=0,
    preprocess="resize_and_crop",
    no_flip=True,
    serial_batches=True,
    batch_size=1,
)


def _build_opt(
    *,
    checkpoints_dir: Path,
    name: str,
    epoch: str,
    gpu_ids: list[int],
) -> SimpleNamespace:
    """Pix2PixModel 이 기대하는 opt 객체를 SimpleNamespace 로 구성."""
    opt = SimpleNamespace(**_TRAIN_OPT_DEFAULTS)
    opt.checkpoints_dir = str(checkpoints_dir)
    opt.name = name
    opt.epoch = epoch
    opt.gpu_ids = list(gpu_ids)
    return opt


class DesmokeEngine:
    """PFAN+SurgiATM 추론 엔진.

    사용 예::

        engine = DesmokeEngine.from_default()
        # frame_bgr: np.ndarray (H,W,3) uint8 BGR (예: cv2.imread 결과)
        clean_bgr = engine.process_bgr_frame(frame_bgr)

    Parameters
    ----------
    checkpoints_dir
        ``<...>/PFAN_Final/270_net_G.pth`` 의 부모의 부모 디렉토리.
        예: ``F:/.../src/PFAN-SurgiATM-PGSA/Checkpoint``.
    name
        체크포인트 하위 폴더 이름. 기본값은 ``"PFAN_Final"`` 로 prepare_checkpoint.py 와 일치.
    epoch
        에폭 라벨. ``"270"`` (파일명 ``270_net_G.pth`` 의 prefix).
    device
        ``"cuda"``, ``"cpu"``, 또는 ``"cuda:0"`` 같은 명시적 디바이스. ``None`` 이면 자동 선택.
    """

    INPUT_SIZE = 256  # 모델이 학습된 해상도 (BatchNorm 통계가 이 분포에 맞춰져 있음)

    def __init__(
        self,
        checkpoints_dir: Path | str,
        *,
        name: str = "PFAN_Final",
        epoch: str = "270",
        device: Optional[str] = None,
    ) -> None:
        # ─ 디바이스 결정 ──────────────────────────────────────────
        # 사용자가 명시 안 했으면 GPU 가능 여부로 자동 판단
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # gpu_ids 는 Pix2PixModel 이 내부에서 DataParallel 결정에 쓰므로
        # CPU 추론이면 빈 리스트, GPU 면 [<index>].
        if self.device.type == "cuda":
            gpu_ids = [self.device.index if self.device.index is not None else 0]
        else:
            gpu_ids = []

        # ─ 체크포인트 경로 검증 ──────────────────────────────────────
        ckpt_dir = Path(checkpoints_dir)
        weight_path = ckpt_dir / name / f"{epoch}_net_G.pth"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"체크포인트를 찾을 수 없습니다: {weight_path}\n"
                f"먼저 `python scripts/prepare_checkpoint.py` 를 실행하세요."
            )

        # ─ 옵션 객체 구성 + 모델 인스턴스화 ─────────────────────────
        opt = _build_opt(
            checkpoints_dir=ckpt_dir,
            name=name,
            epoch=epoch,
            gpu_ids=gpu_ids,
        )
        self._opt = opt

        # Pix2PixModel.__init__ 에서 GradScaler() 를 만드는데, GPU 가 없으면 경고가
        # 뜨지만 동작에는 지장 없음 (train 시에만 사용됨). 무시.
        self.model = Pix2PixModel(opt)
        self.model.setup(opt)   # load_networks(epoch) 자동 호출됨
        self.model.eval()       # BatchNorm 통계 고정 — 필수

        # 가중치 채널 정합성 빠른 자가진단 — 학습 시 input_nc 가 3/4 중 무엇이든
        # 실제 conv 입력 차원은 4 여야 정상이다 (PFAN 내부에서 RGB+D_refined cat).
        # DataParallel 일 수도 아닐 수도 있으니 양쪽 모두 처리.
        try:
            netG = self.model.netG.module if hasattr(self.model.netG, "module") else self.model.netG
            in_ch = netG.model_1[0].weight.shape[1]
            if in_ch != 4:
                # 학습 시 가정과 다름 → 이후 추론에서 cat 차원이 안 맞아 터질 수 있음.
                print(
                    f"[DesmokeEngine][WARN] PFAN model_1 입력 채널이 {in_ch} 입니다. "
                    f"기대값(4)과 달라 추론 시 RuntimeError 가 날 수 있습니다."
                )
        except (AttributeError, IndexError):
            # 구조가 달라졌으면 무시 — 핵심 추론은 그대로 진행
            pass

    # ── 팩토리 ─────────────────────────────────────────────────────
    @classmethod
    def from_default(cls, device: Optional[str] = None) -> "DesmokeEngine":
        """프로젝트의 기본 체크포인트 위치로 엔진 생성."""
        return cls(
            checkpoints_dir=PFAN_ROOT / "Checkpoint",
            device=device,
        )

    # ── 텐서 단위 추론 ──────────────────────────────────────────────
    @torch.no_grad()
    def process_tensor(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """이미 (1,3,256,256) [-1,1] 로 가공된 텐서를 받아 디스모킹 결과 텐서를 반환.

        다른 형식이 들어오면 호출자가 직접 정규화/리사이즈해야 한다.
        """
        if frame_tensor.dim() != 4 or frame_tensor.shape[1] != 3:
            raise ValueError(
                f"입력 텐서 shape 가 (1,3,H,W) 여야 합니다. 받은 shape: {tuple(frame_tensor.shape)}"
            )

        # set_input 이 받는 dict 구조 — B 는 사용 안 되지만 키가 있어야 KeyError 방지.
        self.model.set_input({
            "A": frame_tensor,
            "B": frame_tensor,            # dummy (test 시 D 가 만들어지지 않으므로 안전)
            "A_paths": ["<stream>"],
            "B_paths": ["<stream>"],
        })
        self.model.test()                 # 내부에서 with no_grad: forward()
        return self.model.fake_B          # (1,3,256,256), [-1,1]

    # ── numpy(BGR) 단위 편의 API ───────────────────────────────────
    def process_bgr_frame(self, bgr: np.ndarray) -> np.ndarray:
        """OpenCV BGR uint8 프레임을 받아 디스모킹된 BGR uint8 프레임을 반환.

        - 입력 해상도(H,W) 는 임의여도 됨. 내부적으로 256×256 로 추론한 뒤
          원본 해상도로 다시 업스케일 한다 (모델이 256 분포로 학습됐기 때문).
        - 색공간: cv2 와 동일하게 BGR 입출력 → GUI/녹화에 그대로 쓰기 편함.
        """
        clean, _dcp, _smoke = self.process_bgr_frame_with_maps(bgr, make_maps=False)
        return clean

    # ── 시각화 맵까지 함께 반환 (시연용) ──────────────────────────
    def process_bgr_frame_with_maps(
        self,
        bgr: np.ndarray,
        *,
        make_maps: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """디스모킹 결과 + DCP map + 연기 분포 히트맵을 한 번에 반환.

        Returns
        -------
        clean_bgr : np.ndarray
            디스모킹된 BGR uint8 프레임 (원본 해상도).
        dcp_bgr : np.ndarray | None
            Dark Channel Prior 시각화 (BGR uint8, 원본 해상도). ``make_maps=False`` 시 ``None``.
        smoke_bgr : np.ndarray | None
            연기 분포 히트맵 — 물리 모델이 실제로 빼낸 연기량 ``(η+D)/(η+1)·(1-ρ)`` 시각화.
            ``make_maps=False`` 시 ``None``.

        Notes
        -----
        - 맵 두 장은 단일 채널 [0,1] 텐서를 OpenCV 컬러맵으로 변환한 결과.
          DCP 는 BONE (옅은 → 진한 청회색), Smoke 는 INFERNO (검정 → 노랑) 로 의료 톤에 맞춤.
        - 추가 비용은 그래도 작음(텐서→numpy→resize). GUI fps 영향은 미미.
        """
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("입력 프레임은 (H,W,3) BGR 형식이어야 합니다.")

        import cv2  # 지연 임포트 — process_tensor 단독 사용 시엔 cv2 없이도 OK

        original_h, original_w = bgr.shape[:2]

        # 1) 모델 입력 텐서 준비 (256×256, RGB, [-1,1])
        rgb_small = cv2.cvtColor(
            cv2.resize(bgr, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        tensor = torch.from_numpy(rgb_small).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 127.5 - 1.0
        tensor = tensor.to(self.device)

        # 2) 추론 — forward 가 끝나면 model.D_refined / rho_DNN_norm 이 채워짐
        out_tensor = self.process_tensor(tensor)

        # 3) 디스모킹 결과 BGR 변환 + 원본 해상도로 업스케일
        out_rgb = tensor2im(out_tensor)  # (H,W,3) uint8 RGB
        clean_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        if (original_h, original_w) != (self.INPUT_SIZE, self.INPUT_SIZE):
            clean_bgr = cv2.resize(
                clean_bgr,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )

        if not make_maps:
            return clean_bgr, None, None

        # 4) 내부 텐서 꺼내기 — 모두 (1,1,256,256), [0,1] 범위
        d_refined = self._extract_scalar_map(getattr(self.model, "D_refined", None))
        rho_norm = self._extract_scalar_map(getattr(self.model, "rho_DNN_norm", None))

        # 5) 두 맵을 컬러맵으로 변환 (실패 시 None 으로 안전하게 폴백)
        dcp_bgr: Optional[np.ndarray] = None
        smoke_bgr: Optional[np.ndarray] = None

        if d_refined is not None:
            dcp_bgr = self._colorize_map(
                d_refined,
                out_size=(original_w, original_h),
                colormap=cv2.COLORMAP_BONE,
            )

        if d_refined is not None and rho_norm is not None:
            # 물리 모델이 실제로 빼낸 연기량 — SurgiATM 의 dc_rho 와 동일한 수식
            # eta=0.1 은 SurgiATM 기본값과 동일 (모델 init 에서 변경된 적 없음).
            eta = 0.1
            smoke_density = (eta + d_refined) / (eta + 1.0) * (1.0 - rho_norm)
            # 시각화를 위해 [0,1] 클램프 — 수치 오차로 살짝 음수가 날 수 있어 안전 처리
            smoke_density = np.clip(smoke_density, 0.0, 1.0)
            smoke_bgr = self._colorize_map(
                smoke_density,
                out_size=(original_w, original_h),
                colormap=cv2.COLORMAP_INFERNO,
            )

        return clean_bgr, dcp_bgr, smoke_bgr

    # ── 내부 유틸 ─────────────────────────────────────────────────
    @staticmethod
    def _extract_scalar_map(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        """모델 내부 텐서를 (H,W) float32 numpy 로 환산.

        ──────────────────────────────────────────────────────────
        형태 처리:
        - (1,1,H,W) → squeeze → (H,W)                        ··· DCP map
        - (1,3,H,W) → 채널 평균(luminance 근사) → (H,W)      ··· rho_DNN_norm
        - (1,H,W) / (H,W) → 그대로

        rho_DNN 은 RGB 채널별로 다르게 예측되지만, 시연용 "연기 분포" 는 단일
        스칼라가 더 읽기 쉬워서 채널 평균을 사용한다. 색상 왜곡 정도가 채널마다
        다른 데서 오는 미세 차이는 시각화 톤에선 무시 가능.
        ──────────────────────────────────────────────────────────
        """
        if t is None:
            return None
        arr = t.detach().to("cpu").float().numpy()
        # 배치 차원 제거
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # 채널 차원 정리
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                # 멀티채널은 평균으로 단일채널 환산 (luminance 근사)
                arr = arr.mean(axis=0)
        if arr.ndim != 2:
            return None
        return arr

    @staticmethod
    def _colorize_map(
        scalar: np.ndarray,
        *,
        out_size: tuple[int, int],
        colormap: int,
    ) -> np.ndarray:
        """[0,1] 단일채널 맵을 컬러맵으로 변환 + 원본 해상도로 업스케일.

        Parameters
        ----------
        scalar : (H,W) float32 in [0,1]
        out_size : (W, H) — cv2.resize 가 기대하는 순서
        colormap : cv2.COLORMAP_* 상수
        """
        import cv2
        u8 = (np.clip(scalar, 0.0, 1.0) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(u8, colormap)  # BGR uint8
        if (colored.shape[1], colored.shape[0]) != out_size:
            colored = cv2.resize(colored, out_size, interpolation=cv2.INTER_LINEAR)
        return colored
