import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import io, color, filters
import math
from collections import OrderedDict

try:
    import wandb
except ImportError:
    wandb = None

# Visdom 예외 처리 가이드
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def rmetrics(a, b):
    """지표 계산 로직 (0-255 범위 가정)"""
    mse = np.mean((a - b)**2)
    if mse == 0:
        psnr = 100
    else:
        # float 데이터일 경우 1.0 기준, uint8일 경우 255 기준 확인 필요
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    # channel_axis는 skimage 버전에 따라 다를 수 있음
    ssim = compare_ssim(a, b, multichannel=True, channel_axis=2)
    return mse, psnr, ssim

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        # 옵션에 없거나 1이어도 강제로 0으로 취급하게 하거나, 안전하게 가져옵니다.
        self.display_id = getattr(opt, 'display_id', 0)
        self.use_html = opt.isTrain and not getattr(opt, 'no_html', False)
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = getattr(opt, 'display_port', 8097)
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        # [수정] Visdom 연결 시도 자체를 'try-except'로 격리
        if self.display_id > 0:
            try:
                import visdom
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
                if not self.vis.check_connection():
                    self.create_visdom_connections()
            except (ImportError, ModuleNotFoundError):
                print("⚠️ Warning: 'visdom' package not found. Skipping Visdom visualization.")
                self.display_id = 0  # 라이브러리가 없으면 ID를 0으로 강제 전환
            except Exception as e:
                print(f"⚠️ Warning: Visdom connection failed: {e}. Skipping.")
                self.display_id = 0

        # Visdom 초기화 (display_id > 0 일 때만 실행)
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # WandB 초기화
        if self.use_wandb and wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run

        # 디렉토리 및 로그 파일 생성
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.log_pfm_val_name = os.path.join(opt.checkpoints_dir, opt.name, 'Val_Performance_log.txt')
        
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ Training Loss ({time.strftime("%c")}) ================\n')
        with open(self.log_pfm_val_name, "a") as log_pfm_val_file:
            log_pfm_val_file.write(f'================ Val_Performance ({time.strftime("%c")}) ================\n')

    def reset(self):
        self.saved = False

    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\nCould not connect to Visdom server. Trying to start a server...')
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """이미지 시각화 및 WandB 업로드"""
        # 1. Visdom 출력 (ID가 있을 때만)
        if self.display_id > 0:
            title = self.name
            images = []
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                images.append(image_numpy.transpose([2, 0, 1]))
            try:
                self.vis.images(images, nrow=self.ncols, win=self.display_id + 1,
                                opts=dict(title=title + ' images'))
            except VisdomExceptionBase:
                self.create_visdom_connections()

        # 2. WandB 출력 (핵심)
        if self.use_wandb and wandb:
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                ims_dict[f"Visuals/{label}"] = wandb.Image(image_numpy)
            wandb.log(ims_dict, step=epoch)

        # 3. HTML 저장
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """학습 Loss 그래프 출력"""
        # Visdom
        if self.display_id > 0:
            if not hasattr(self, 'plot_data'):
                self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
            self.plot_data['X'].append(epoch + counter_ratio)
            self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
            try:
                self.vis.line(
                    X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                    Y=np.array(self.plot_data['Y']),
                    opts={'title': self.name + ' loss', 'legend': self.plot_data['legend'], 'xlabel': 'epoch', 'ylabel': 'loss'},
                    win=self.display_id)
            except VisdomExceptionBase:
                self.create_visdom_connections()

        # WandB (Loss는 'Loss/' 접두어를 붙여 깔끔하게 관리)
        if self.use_wandb and wandb:
            wandb.log({f"Loss/{k}": v for k, v in losses.items()}, step=epoch)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def cal_current_pfm(self, epoch, visuals):
        """성능 지표 계산 로직 (Visdom 독립적)"""
        # visuals dict에서 0: Input, 1: Fake, 2: Real(GT) 순서라고 가정
        images_list = []
        for label, image in visuals.items():
            images_list.append(util.tensor2im(image))
        
        # 안전한 인덱싱 (최소 3개 이미지가 있다고 가정)
        im_out = images_list[1]
        im_GT = images_list[2]
        
        mse, psnr, ssim = rmetrics(im_out, im_GT)
        return OrderedDict([('MSE', mse), ('PSNR', psnr), ('SSIM', ssim)])

    def plot_current_ssim_val(self, epoch, counter_ratio, mt_pfm):
        """검증 지표 시각화"""
        # Visdom (생략 가능하나 유지)
        if self.display_id > 0:
            try:
                # MSE, PSNR, SSIM 각각 개별 창에 그리는 기존 로직 유지 가능
                pass 
            except: pass

        # WandB (검증 지표 로그)
        if self.use_wandb and wandb:
            wandb.log({f"Val/{k}": v for k, v in mt_pfm.items()}, step=epoch)

    def print_current_val_mtx(self, epoch, t_comp, t_data, mtpfm):
        message_pfm = '(VAL epoch: %d, time: %.3f, data: %.3f) ' % (epoch, t_comp, t_data)
        for k, v in mtpfm.items():
            message_pfm += '%s: %.3f ' % (k, v)
        print(message_pfm)
        with open(self.log_pfm_val_name, "a") as log_pfm_val_file:
            log_pfm_val_file.write('%s\n' % message_pfm)