import torch
from .base_model import BaseModel
from . import networks
from torch.cuda.amp import GradScaler
from models.SurgiATM import SurgiATM, guided_filter
from pytorch_msssim import ms_ssim, ssim

def edge_aware_smoothness_loss(rho_DNN, img, alpha=10.0):
    """
    img: [B, C, H, W] 원본 이미지
    rho_DNN: [B, 1, H, W] 예측된 물리 파라미터 맵
    """
    img_gray = img.mean(dim=1, keepdim=True)
    
    # 가로, 세로 방향의 Gradient(엣지 강도) 계산
    grad_rho_x = torch.abs(rho_DNN[:, :, :, :-1] - rho_DNN[:, :, :, 1:])
    grad_rho_y = torch.abs(rho_DNN[:, :, :-1, :] - rho_DNN[:, :, 1:, :])
    grad_img_x = torch.abs(img_gray[:, :, :, :-1] - img_gray[:, :, :, 1:])
    grad_img_y = torch.abs(img_gray[:, :, :-1, :] - img_gray[:, :, 1:, :])
    
    # 원본에 엣지가 있으면(grad_img가 크면) exp 값은 0에 수렴 -> rho의 변화 허용
    # 원본이 평탄하면(grad_img가 작으면) exp 값은 1에 수렴 -> rho의 변화를 0으로 억제(평탄화)
    loss_x = grad_rho_x * torch.exp(-alpha * grad_img_x)
    loss_y = grad_rho_y * torch.exp(-alpha * grad_img_y)
    
    return loss_x.mean() + loss_y.mean()

class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(
            netG='pfan',           # 기본 생성기를 PFAN으로 설정 [cite: 7, 8]
            netD='basic',          # 기본 판별기를 PatchGAN으로 설정 
            dataset_mode='aligned', # 사용자의 폴더 구조(input/target 분리)에 맞춤
            norm='batch',          # 논문에서 사용한 BatchNorm
            beta1=0.5,             # 논문의 Adam 옵티마이저 설정 [cite: 193]
            lr=0.0002,             # 논문의 기본 학습률 [cite: 193]
            lambda_L1=100.0        # 논문의 L1 손실 가중치
        )
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--surgiatm_wz', type=int, default=15, help='SurgiATM dark channel window size')
            parser.add_argument('--lambda_smooth', type=float, default=0.1, help='Edge-Aware Smoothness Loss Param')
            parser.add_argument('--smooth_alpha', type=float, default=10.0, help='SurgiATM dark channel window size')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_smooth', 'G_OOB', 'G_SSIM', 'D_real', 'D_fake']
        
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt,opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(3 + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 0.25, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        self.lambda_L1 = opt.lambda_L1       # 보통 100.0
        self.lambda_SSIM = opt.lambda_L1     # SSIM도 100.0 수준으로 주어 강력하게 강제
        self.surgiatm = SurgiATM(dc_window_size=opt.surgiatm_wz).to(self.device)
        self.scaler = GradScaler()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        with torch.autocast(device_type='cuda'):    
            # 1. [추가] Input 단에서 물리적 사전지식(Dark Channel) 선제 추출
            self.real_A_norm = (self.real_A + 1.0) / 2.0
            with torch.no_grad():
                # SurgiATM의 헬퍼 함수를 호출하여 D_refined 획득
                dc_coarse = self.surgiatm.get_dc(self.real_A_norm)
                guide_I = self.real_A_norm.mean(dim=-3, keepdim=True)
                self.D_refined = guided_filter(guide_I, dc_coarse, radius=self.opt.surgiatm_wz // 2).detach()
            
            self.rho_DNN = self.netG(self.real_A, self.D_refined)
            
            
            # 1. SurgiATM 물리 수식 통과
            # 물리 연산을 위해 [0, 1] 공간으로 매핑된 텐서 생성 (메모리 포인터 분리)
            self.rho_DNN_norm = (self.rho_DNN + 1.0) / 2.0
            pre_clean_image = self.surgiatm(smoky_image = self.real_A_norm, 
                                            rho_DNN = self.rho_DNN_norm,
                                            precomputed_D = self.D_refined)
            
            # 2. 판별기 붕괴 방지를 위한 Forward Clamping 및 Out-of-bound 패널티 보존
            self.pre_clean_image = pre_clean_image # Loss 계산을 위해 원본 보존
            pre_clean_clamped = torch.clamp(pre_clean_image, min=0.0, max=1.0)
            
            # 3. 최종 출력 (판별기로 전달)
            self.fake_B = (pre_clean_clamped * 2.0) - 1.0

    def backward_D(self):
        with torch.autocast(device_type='cuda'):
            """Calculate GAN loss for the discriminator"""
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.scaler.scale(self.loss_D).backward()

    def backward_G(self):
        with torch.autocast(device_type='cuda'):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            
            ssim_val = ssim(self.fake_B, self.real_B, data_range=2.0, size_average=True)
            self.loss_G_SSIM = (1.0 - ssim_val) * self.lambda_SSIM

            # [수정 3 적용] OOB Loss를 클래스 변수로 할당하여 로깅 가능하게 처리
            self.loss_G_OOB = torch.mean(torch.relu(-self.pre_clean_image) + torch.relu(self.pre_clean_image - 1.0)) * 10.0
            
            # [수정 1, 2 적용] 논리적 가이드(real_B) 사용 & 물리적 범위([0, 1]) 매핑
            # 훈련 시에는 연기가 없는 정답 이미지(real_B)를 가이드로 사용하여 완벽한 구조적 엣지만 학습 강제
            real_B_norm = (self.real_B + 1.0) / 2.0
            self.loss_G_smooth = edge_aware_smoothness_loss(self.rho_DNN_norm, real_B_norm, self.opt.smooth_alpha) * self.opt.lambda_smooth

            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SSIM + self.loss_G_OOB + self.loss_G_smooth

        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self, epoch):
        self.forward()                   # compute fake images: G(A) -> SurgiATM 적용
        
        # update D
        self.set_requires_grad(self.netD, True)  
        self.optimizer_D.zero_grad()     
        self.backward_D()                # calculate gradients for D (Scaled)
        self.scaler.step(self.optimizer_D) 
        
        # update G
        self.set_requires_grad(self.netD, False)  
        self.optimizer_G.zero_grad()        
        self.backward_G()                   # calculate graidents for G (Scaled)
        self.scaler.step(self.optimizer_G)  

        self.scaler.update()
