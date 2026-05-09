import torch
from torch import nn
import torch.nn.functional as F

# [추가] O(N) 선형 복잡도를 가지는 Differentiable Guided Filter 구현
def guided_filter(guide, src, radius, eps=1e-3):
    kernel_size = 2 * radius + 1
    # Mean 연산은 O(N)의 AvgPool2d로 대체하여 메모리 접근 최적화
    mean_I = F.avg_pool2d(guide, kernel_size, stride=1, padding=radius)
    mean_p = F.avg_pool2d(src, kernel_size, stride=1, padding=radius)
    mean_Ip = F.avg_pool2d(guide * src, kernel_size, stride=1, padding=radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = F.avg_pool2d(guide * guide, kernel_size, stride=1, padding=radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = F.avg_pool2d(a, kernel_size, stride=1, padding=radius)
    mean_b = F.avg_pool2d(b, kernel_size, stride=1, padding=radius)

    return mean_a * guide + mean_b

class SurgiATM(nn.Module):
    def __init__(self, dc_window_size=15, eta=0.1, w=0.95):
        super().__init__()
        self.dc_wz = dc_window_size
        self.w = w
        self.eta = eta

    def get_dc(self, x: torch.Tensor):
        pd = self.dc_wz // 2
        dc_pixel_wise, _ = x.min(dim=-3, keepdim=True)
        if self.dc_wz > 1:
            dc_pixel_wise_pad = F.pad(dc_pixel_wise, (pd, pd, pd, pd), mode="replicate")
            dc = - F.max_pool2d(-dc_pixel_wise_pad, kernel_size=self.dc_wz, stride=1, padding=0)
        else:
            dc = dc_pixel_wise
        return dc
    
    def forward(self, smoky_image: torch.Tensor, rho_DNN: torch.Tensor, precomputed_D: torch.Tensor = None):
        # 1. Zero-Overhead Fast Guided Filter (우리의 공간 최적화 유지 - Halo 방지)
        with torch.no_grad():
            if precomputed_D is not None:
                dc_refined = precomputed_D
            else:
                dc_coarse = self.get_dc(smoky_image)
                guide_I = smoky_image.mean(dim=-3, keepdim=True)
                dc_refined = guided_filter(guide_I, dc_coarse, radius=self.dc_wz//2, eps=1e-3)
            
            # VRAM 누수 차단
            dc_refined = dc_refined.detach() 

        # 2. 원본 SurgiATM의 수학적 논리 복원 (A 상수 및 나눗셈 영구 삭제)
        # 나눗셈이 사라졌으므로 t_map_safe나 Softplus 같은 거추장스러운 방어 코드가 필요 없음
        dc_rho = (self.eta + dc_refined) / (self.eta + 1.0) * (1 - rho_DNN)

        # 3. 뺄셈 기반의 안전한 물리 렌더링 (Gradient 폭발 및 푸른색 색상 왜곡 원천 차단)
        pre_clean_image = smoky_image - dc_rho
        
        return pre_clean_image