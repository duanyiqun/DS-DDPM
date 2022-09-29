"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) evaluation/sampling
summary: >
  Code to generate samples from a trained
  Denoising Diffusion Probabilistic Model.
---

# [Denoising Diffusion Probabilistic Models (DDPM)](index.html) evaluation/sampling

This is the code to generate images and create interpolations between given images.
"""
from multiprocessing import reduction
from turtle import Turtle
from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from EEGNet import EEG_Net_8_Stack
from unet_eeg_subject_emb import sub_gaussion
from unet_eeg import UNet
from typing import Tuple, Optional

from cgi import test
from threading import stack_size
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize
from typing import Optional, Tuple

from labml import experiment, monit
# from labml_nn.diffusion.ddpm import DenoiseDiffusion, gather
from utils import gather
# from diffusion_uncon_bci_compiv import Configs
from comon_two_stream_sep_gaussion_organal_arc_margin import Configs
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from utils import gather


import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from utils import gather
import numpy as np
import scipy.io as sio
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import os
import mne
import scipy.io as sio
# from mudus.visualization.visualize_utils_mne import raw_construct_mne_info
from visualization.visualize_utils_mne import raw_construct_mne_info
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from scipy.interpolate import griddata
from PIL import Image
import gif

# subject_instance = UNet_sub()

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Parameter 的用途：
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的
        # https://www.jianshu.com/p/d8b77cc02410
        # 初始化权重
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ArcMarginHead(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, load_backbone = '/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/logs/subjects_normal/BCI_IV_EEGNet_batchsize256_lr0.05_gamma0.5_step20_maxepoch100_exp1/max_acc.pth', s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginHead, self).__init__()
        self.arcpro = ArcMarginProduct(in_features, out_features, s=30.0, m=0.50, easy_margin=False)
        self.auxback = EEG_Net_8_Stack(mtl=False)
        pretrained_checkpoint = torch.load(load_backbone)
        print("loading the pretrained subject EEGNet weight and convert to arc..{}".format(pretrained_checkpoint.keys()))
        self.auxback.load_state_dict(pretrained_checkpoint['params'])
        print("backbone loading successful")

        

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        emb = self.auxback(input)
        # print(output)
        output = self.arcpro(emb, label)

        return output


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, sub_theta: nn.Module, sub_arc_head: nn.Module, debug=False):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        self.sub_theta = sub_theta
        self.sub_arc_head = sub_arc_head

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta
        self.debug = debug
        self.step_size = 75
        self.window_size = 224
        # self.step_size = 93.5
        # self.window_size = 93.5
        self.subject_noise_range = 9
        # self.arcmargin = ArcMarginProduct()

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        if self.debug:
            print("the selected alpha bar would be {}".format(gather(self.alpha_bar, t).shape))
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps


    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def p_sample_noise(self, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.sub_theta(xt, t, s)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, debug=False):
        """
        #### Simplified Loss

        $$L_simple(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        if debug:
            print("the shape of x0")
            print(x0.shape)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if debug:
            print("the shape of t")
            print(t.shape)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)
            if debug:
                print("the shape of noise")
                print(noise.shape)
        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        if debug:
            print("the shape of xt")
            print(xt.shape)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)
        # MSE loss
        return F.mse_loss(noise, eps_theta)

    
    def loss_with_diff_constraint(self, x0: torch.Tensor, label: torch.Tensor, 
                                    noise: Optional[torch.Tensor] = None, debug=False, 
                                    noise_content_kl_co = 1, arc_subject_co = 0.1):
        """
        #### Simplified Loss

        $$L_simple(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        debug = self.debug
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        if debug:
            print("the shape of x0")
            print(x0.shape)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        # s = torch.randint(0, self.subject_noise_range, (batch_size,), device=x0.device, dtype=torch.long)
        s = label
        if debug:
            #  注意这里t采样是0-1000的 随机数，就生成不需要完全每个时刻顺序采，只要随机对上每个t时刻的就可以
            print("the shape of t")
            print(t.shape)
            print(t)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)
            if debug:
                print("the shape of noise")
                print(noise.shape)
        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        if debug:
            print("the shape of xt")
            print(xt.shape)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)
        subject_mu, subject_theta = self.sub_theta(xt, t, s)
        # print(subject_theta.shape)
        if debug:
            print("the shape of eps_theta")
            print(eps_theta.shape)
        constraint_panelty = 0
        
        # eps_layernorm = F.layer_norm(eps_theta, [eps_theta.shape[1], eps_theta.shape[2], eps_theta.shape[3]])
        for i in range(eps_theta.shape[3] - 1):
            # i * self.step_size, i * self.step_size + self.window_size
            # (i + 1) * self.step_size,  (i + 1) * self.step_size + self.window_size
            if debug:
                print("logging with constraint panelty value")
                # print(F.mse_loss(eps_layernorm[:, :, (i + 1) * self.step_size : i * self.step_size + self.window_size, i], eps_layernorm[:, :, (i + 1) * self.step_size : i * self.step_size + self.window_size, i + 1], reduction='none').shape)
                # print(F.mse_loss(eps_layernorm[:, :, self.step_size:, i], eps_layernorm[:, :, :-self.step_size, i + 1], reduction='none').shape)
                # print(F.mse_loss(eps_layernorm[:, :, self.step_size:, i], eps_layernorm[:, :, :-self.step_size, i + 1], reduction='mean'))
                print(F.mse_loss(eps_theta[:, :, self.step_size:, i], eps_theta[:, :, :-self.step_size, i + 1], reduction='mean'))
            # constraint_panelty = constraint_panelty +F.mse_loss(eps_layernorm[:, :, self.step_size:, i], eps_layernorm[:, :, :-self.step_size, i + 1], reduction='mean')
            constraint_panelty = constraint_panelty +F.mse_loss(eps_theta[:, :, self.step_size:, i], eps_theta[:, :, :-self.step_size, i + 1], reduction='mean')
        if debug:
            print("logging with constraint panelty value")
            print(constraint_panelty)

        # MSE loss
        # return F.mse_loss(noise, eps_theta + subject_theta) + constraint_panelty, constraint_panelty
        # noise_conent_kl = F.kl_div(eps_theta.softmax(dim=-1).log(), subject_theta.softmax(dim=-1), reduction='sum')
        noise_conent_kl = F.kl_div(eps_theta.softmax(dim=-1).log(), subject_theta.softmax(dim=-1), reduction='mean')
        subject_arc_logit = self.sub_arc_head(subject_theta.permute(0,3,2,1), s)
        subject_arc_loss = F.cross_entropy(subject_arc_logit, s.long())
        # print('arcloss is {}'.format(subject_arc_loss))
        # return F.mse_loss(noise, eps_theta + subject_theta) + 0.01 * 1/noise_conent_kl, constraint_panelty, noise_conent_kl
        return F.mse_loss(noise, eps_theta + subject_theta) - noise_content_kl_co * noise_conent_kl + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss
        # return F.mse_loss(noise, eps_theta), constraint_panelty


class Sampler:
    """
    ## Sampler class
    """

    def __init__(self, diffusion: DenoiseDiffusion, image_channels: int, window_size: int, step: int, stack_size: int, device: torch.device):
        """
        * `diffusion` is the `DenoiseDiffusion` instance
        * `image_channels` is the number of channels in the image
        * `device` is the device of the model
        """
        self.device = device
        # self.image_size = image_size
        self.image_channels = image_channels
        self.diffusion = diffusion

        self.window_size = window_size
        self.stack_size = stack_size
        self.step = step

        # $T$
        self.n_steps = diffusion.n_steps
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        self.eps_model = diffusion.eps_model
        self.subnoise_model = diffusion.sub_theta
        # $\beta_t$
        self.beta = diffusion.beta
        # $\alpha_t$
        self.alpha = diffusion.alpha
        # $\bar\alpha_t$
        self.alpha_bar = diffusion.alpha_bar
        # $\bar\alpha_{t-1}$
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]])

        # To calculate
        #
        # \begin{align}
        # q(x_{t-1}|x_t, x_0) &= \mathcal{N} \Big(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I} \Big) \\
        # \tilde\mu_t(x_t, x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #                          + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
        # \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{a}
        # \end{align}

        # $\tilde\beta_t$
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        # $$\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$$
        self.mu_tilde_coef1 = self.beta * (alpha_bar_tm1 ** 0.5) / (1 - self.alpha_bar)
        # $$\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1}}{1-\bar\alpha_t}$$
        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

    def show_image(self, img, title=""):
        """Helper function to display an image"""
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(title)
        plt.show()
        data_path = '/home/yiqduan/Data/ddpm/ddpm/logs/diffuse/visualize_celebA/{}.png'.format(title)
        plt.savefig(data_path)

    def make_video(self, frames, path="video.mp4"):
        """Helper function to create a video"""
        import imageio
        # 20 second video
        writer = imageio.get_writer(path, fps=len(frames) // 20)
        # Add each image
        for f in frames:
            f = f.clip(0, 1)
            f = to_pil_image(resize(f, [368, 368]))
            writer.append_data(np.array(f))
        #
        writer.close()

    def sample_animation(self, n_frames: int = 1000, create_video: bool = False, overlap_cover = True, debug=True, subject_id = [5]):
        """
        #### Sample an image step-by-step using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        We sample an image step-by-step using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$ and at each step
        show the estimate
        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        """

        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([1, self.image_channels, self.window_size, self.stack_size], device=self.device)
        print('shape of xt is {}'.format(xt.shape))
        s = torch.tensor(subject_id, dtype=torch.long, device=self.device)
        # Interval to log $\hat{x}_0$
        interval = self.n_steps // n_frames
        # Frames for video
        frames = []
        # Sample $T$ steps
        real_cureves_list = []
        real_noises_list = []
        for t_inv in monit.iterate('Denoise', self.n_steps):
            # $t$
            t_ = self.n_steps - t_inv - 1
            # $t$ in a tensor
            t = xt.new_full((1,), t_, dtype=torch.long)
            # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
            eps_theta = self.eps_model(xt, t)
            if debug:
                print("validating shape of t and s")
                print(t.shape)
                print(s.shape)
            _, subnoise_theta = self.subnoise_model(xt,t,s)
            if debug:
                print("validating shape of noise and eps")
                print(subnoise_theta.shape)
                print(eps_theta.shape)
            real_waves_list = []
            real_noise_list = []
##################################################
            if t_ % interval == 0:
                # Get $\hat{x}_0$ and add to frames
                x0 = self.p_x0(xt, t, eps_theta)
                x0_noise = self.p_x0(xt, t, subnoise_theta)
                
                # frames.append(x0[0])
                x0 = x0.detach().cpu().numpy()
                x0_noise = x0_noise.detach().cpu().numpy()
                real_waves = np.ones((x0.shape[0], 22, 750))
                real_noises = np.ones((x0.shape[0], 22, 750))
                if overlap_cover:
                    for i in range(x0.shape[3]):
                        if debug:
                            print("{}-{}".format(0 + i* self.step, self.window_size + i * self.step))
                        real_waves[:,:,0 + i* self.step: self.window_size + i * self.step] = x0[:,:,:,i]
                        real_noises[:,:,0 + i* self.step: self.window_size + i * self.step] = x0_noise[:,:,:,i]
                else:
                    for i in range(x0.shape[3]):
                        if debug:
                            print("{}-{}".format(0 + i* self.step, self.window_size + i * self.step))
                        real_waves[:,:,0 + i* self.step: self.window_size + i * self.step] = x0[:,:,:,i]
                        real_noises[:,:,0 + i* self.step: self.window_size + i * self.step] = x0_noise[:,:,:,i]
                real_waves_list.append(real_waves)
                real_noise_list.append(real_noises)
                dataroot='/home/yiqduan/Data/ddpm/ddpm/visualization/generated_subject_noise/subject_1'
                curve_temp_path = "{}/curve_{}.png".format(dataroot, t_)
                noise_temp_path = "{}/noise_{}_subject_{}.png".format(dataroot, t_, subject_id)
                # curve_temp_path = "{}/curve_channel1{}.png".format(dataroot, t_)
                plt.clf()
                annimate_curve_change(test_X = real_waves[0:,0:6,:400], given_path=curve_temp_path, color='b')
                annimate_curve_change(test_X = real_noises[0:,0:6,:400], given_path=noise_temp_path, color='r')
                real_cureves_list.append(Image.open(curve_temp_path))
                real_noises_list.append(Image.open(noise_temp_path))
            
                # print(x0.shape)
                # if not create_video:
                    # self.show_image(x0[0], f"{t_}")
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt_noise = self.p_sample(xt, t, subnoise_theta)
            # xt_noise = F.normalize(xt_noise, p=2, dim=2)
            xt = F.normalize(self.p_sample(xt, t, eps_theta) + xt_noise, p=2, dim=2)
            # xt = self.p_sample(xt, t, eps_theta)
            
        # current_path = "{}/curve_animate_content{}.gif".format(dataroot, t_inv)
        # current_noise_path = "{}/noise_curve_animate_subject_{}.gif".format(dataroot, subject_id ,t_inv)
        # fig.colorbar(surf, shrink=0.2, aspect=5)
        gif.save(real_cureves_list, current_path)
        gif.save(real_noises_list, current_noise_path)
                
        # visualize_animate_surface(test_X=real_waves_list[-1], test_y=None, given_path='/home/yiqduan/Data/ddpm/ddpm/visualization/bci_comp4/')

        # Make video
        if create_video:
            self.make_video(frames)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, lambda_: float, t_: int = 100):
        """
        #### Interpolate two images $x_0$ and $x'_0$

        We get $x_t \sim q(x_t|x_0)$ and $x'_t \sim q(x'_t|x_0)$.

        Then interpolate to
         $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$

        Then get
         $$\bar{x}_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|\bar{x}_t)$$

        * `x1` is $x_0$
        * `x2` is $x'_0$
        * `lambda_` is $\lambda$
        * `t_` is $t$
        """

        # Number of samples
        n_samples = x1.shape[0]
        # $t$ tensor
        t = torch.full((n_samples,), t_, device=self.device)
        # $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$
        xt = (1 - lambda_) * self.diffusion.q_sample(x1, t) + lambda_ * self.diffusion.q_sample(x2, t)

        # $$\bar{x}_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|\bar{x}_t)$$
        return self._sample_x0(xt, t_)

    def interpolate_animate(self, x1: torch.Tensor, x2: torch.Tensor, n_frames: int = 100, t_: int = 100,
                            create_video=True, debug=False):
        """
        #### Interpolate two images $x_0$ and $x'_0$ and make a video

        * `x1` is $x_0$
        * `x2` is $x'_0$
        * `n_frames` is the number of frames for the image
        * `t_` is $t$
        * `create_video` specifies whether to make a video or to show each frame
        """

        # Show original images
        self.show_image(x1, "x1")
        self.show_image(x2, "x2")
        # Add batch dimension
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        # $t$ tensor
        t = torch.full((1,), t_, device=self.device)
        # $x_t \sim q(x_t|x_0)$
        x1t = self.diffusion.q_sample(x1, t)
        # $x'_t \sim q(x'_t|x_0)$
        x2t = self.diffusion.q_sample(x2, t)

        frames = []
        # Get frames with different $\lambda$
        for i in monit.iterate('Interpolate', n_frames + 1, is_children_silent=True):
            # $\lambda$
            lambda_ = i / n_frames
            # $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$
            xt = (1 - lambda_) * x1t + lambda_ * x2t
            # $$\bar{x}_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|\bar{x}_t)$$
            x0 = self._sample_x0(xt, t_)
            # Add to frames
            frames.append(x0[0])
            # Show frame
            if not create_video:
                self.show_image(x0[0], f"{lambda_ :.2f}")

        # Make video
        if create_video:
            self.make_video(frames)

    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        """
        #### Sample an image using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        * `xt` is $x_t$
        * `n_steps` is $t$
        """

        # Number of sampels
        n_samples = xt.shape[0]
        # Iterate until $t$ steps
        for t_ in monit.iterate('Denoise', n_steps):
            t = n_steps - t_ - 1
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt = self.diffusion.p_sample(xt, xt.new_full((n_samples,), t, dtype=torch.long))

        # Return $x_0$
        return xt

    def sample(self, n_samples: int = 16):
        """
        #### Generate images
        """
        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

        # $$x_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|x_t)$$
        x0 = self._sample_x0(xt, self.n_steps)

        # Show images
        # for i in range(n_samples):
        #     self.show_image(x0[i])

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def p_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        #### Estimate $x_0$

        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        """
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)

        # $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
        #  \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        return (xt - (1 - alpha_bar) ** 0.5 * eps) / (alpha_bar ** 0.5)


def visualize_animate_surface(test_X = None, test_y = None, given_path = None):
    if test_X is None:
        test_X = np.ones((8,22,750))
    if test_y is None:
        test_y = np.ones((test_X.shape[0],))
    custom_epochs = raw_construct_mne_info(test_X, test_y, visual_len=200)
    # print(custom_epochs)
    # custom_epochs['Left'].average().plot(time_unit='s')
    # custom_epochs.plot_image(1, cmap='interactive', sigma=1., vmin=-250, vmax=250)
    # custom_epochs.plot(scalings='auto')
    events = mne.pick_events(custom_epochs.events, include=[1])
    # mne.viz.plot_events(events)
    # plt.show()
    # custom_epochs.plot(events=events, event_color={1: 'r', 2: 'b', 3: 'y', 4: 'g'}, title='Raw trail plot',
    #                    scalings=30, block=False, n_epochs=5, n_channels=11, epoch_colors=None, butterfly=False)
    # here epoch colors is with shape epochs *  channels
    # custom_epochs.plot_image(title='Epoch value time series wise', sigma=1., cmap='YlGnBu_r')    # here epoch colors is with shape epochs *  channels
    # mne.viz.plot_epochs_image(custom_epochs,cmap='interactive',title='Whole data heat map of epochs')
    # custom_epochs['Left'].average().plot(time_unit='s')
    # custom_epochs['Right'].average().plot(time_unit='s')

    evoked = custom_epochs.average()
    info = custom_epochs.info
    info.set_montage('standard_1020')
    # print(evoked.data.shape)
    # print(info)
    # print(info['chs'][0])
    location_dict_1020 = {}
    xp_1020 = []
    yp_1020 = []
    zp_1020 = []
    for itm in info['chs']:
        # print(itm)
        print('the channel name is {} the location is {}'.format(itm['ch_name'], itm['loc'][:3]))
        location_dict_1020[itm['ch_name']] = itm['loc'][:3]
        xp_1020.append(itm['loc'][0])
        yp_1020.append(itm['loc'][1])
        zp_1020.append(itm['loc'][2])
        
    xp_64 = []
    yp_64 = []
    zp_64 = []
    info.set_montage('biosemi64')
    location_dict_64 = {}
    for itm in info['chs']:
        # print(itm)
        print('the channel name is {} the location is {}'.format(itm['ch_name'], itm['loc'][:3]))
        location_dict_64[itm['ch_name']] = itm['loc'][:3]
        xp_64.append(itm['loc'][0])
        yp_64.append(itm['loc'][1])
        zp_64.append(itm['loc'][2])


    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xp_1020, yp_1020, zp_1020, c='r', marker = "^", label='10-20system')
    ax.scatter(xp_64, yp_64, zp_64, c='g', marker="*", label='64-system')
    plt.savefig('/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/mudus/visualization/montage.png')
    fig = plt.figure()
    plt.scatter(xp_1020, yp_1020,  c='r', label='10-20system')
    plt.scatter(xp_64, yp_64,  c='g', label='64-system')
    plt.savefig('/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/mudus/visualization/montage_2d.png')
    print(test_X.shape)
    for idx in range(test_X.shape[0]):
        if idx > 100:
            break
        # print(test_X[idx].shape)
        # print(test_y[idx])
        images = []
        """
        fig = plt.figure()
        ax.set_aspect('auto')
        ax = fig.gca(projection='3d')
        # set figure information
        ax.set_title("trace of activation in spatial wise {}".format(events_id[str(int(test_y[idx]))]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("activation")
        """
        for step in range(len(test_X[idx][1])):
            # print(step)

            fig = plt.figure()
            ax.set_aspect('auto')
            ax = fig.gca(projection='3d')
            # set figure information
            ax.set_title("trace of activation in spatial wise {}".format(events_id[str(int(test_y[idx]))]))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("activation")
            plt.gca().set_box_aspect((1,1,0.5))

            indices = [n for n in range(len(xp_1020))]
            # print(len(indices))
            
            # activation_points = test_X[idx][indices][:]
            # print(test_X[idx].shape)
            # activation_points = standardization(test_X[idx][:, step])
            activation_points = test_X[idx][:, step]
            
            # print(len(activation_points))
            # print(activation_points.shape)
            act_x = [xp_1020[i] for i in indices]
            act_y = [yp_1020[i] for i in indices]
            meshx, meshy = np.meshgrid(act_x, act_y)
            # print(meshx.shape)
            # print(meshy.shape)
            # print("mesh established")
            # ax.set_aspect('auto')
            # ax = fig.gca(projection='3d')
            # set figure information
            # ax.set_title("trace of activation in spatial wise {}".format(events_id[str(int(test_y[idx]))]))
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("activation")
            grid_activation = griddata((act_x, act_y), activation_points, (meshx, meshy), method='nearest')

            # Plot the surface.
            surf = ax.plot_surface(meshx, meshy, grid_activation, cmap=plt.cm.coolwarm,
                                linewidth=0, antialiased=False, animated=True)
            fig.colorbar(surf, shrink=0.2, aspect=5)
            # surf = ax.plot_trisurf(act_x, act_y, activation_points, linewidth=0.2, antialiased=True)
            # Customize the z axis.
            # ax.set_zlim(-2.00, 2.00)
            # ax.set_xlim(-0.075, 0.075)
            # ax.set_xlim(-0.10, 0.10)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # A StrMethodFormatter is used automatically
            # ax.zaxis.set_major_formatter('{x:.02f}')
            # Add a color bar which maps values to colors.
            
            # first_2000 = data1[:, 3
            # second_2000 = data1[:, 7]
            # third_2000 = data1[:, 11]
            # print(len(act_x))
            # figure1 = ax.plot(act_x, act_y, activation_points, c='r', linewidth=0.5)
            current_path = "{}/surface_graph_id_{}.png".format(given_path, "temp")
            # current_path = "{}/surface_graph_id_{}.png".format(given_path, idx)
            plt.savefig(current_path)
            # ax.set_aspect('auto')
            # 
            # im = plt.imshow(surf, animated=True)
            # im = plt.show()
            images.append(Image.open(current_path))
        current_path = "{}/surface_graph_id_{}.gif".format(given_path, idx)
        # fig.colorbar(surf, shrink=0.2, aspect=5)
        gif.save(images, current_path)
        # animate = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
        # animate.save("/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/mudus/visualization/surface_interpolate_animate/surface_graph_id_{}.gif".format(idx))


def annimate_curve_change(test_X = None, test_y = None, given_path = None, color='b'):
    if test_X is None:
        test_X = np.ones((1, 22,750))
    if test_y is None:
        test_y = np.ones(1,)
    # print(test_X.shape)
    t = np.arange(0, test_X.shape[2], 1)
    for channel in range(test_X.shape[1]):
        ax1 = plt.subplot(test_X.shape[1], 1, channel+1)
        plt.plot(t, test_X[0, channel, :], color=color, linewidth=0.3)
        plt.setp(ax1.get_xticklabels(), fontsize=2)
        plt.xlim(1, test_X.shape[2])
        # plt.show()
    current_path = given_path
    plt.savefig(current_path)


def annimate_curve_change_with_diff(test_X = None, test_y = None, given_path = None):
    # plt.cla()
    if test_X is None:
        test_X = np.ones((1, 22, 400, 8))
    if test_y is None:
        test_y = np.ones(1,)
    print(test_X.shape)
    for channel in range(test_X.shape[1]):
        ax1 = plt.subplot(test_X.shape[1], 1, channel+1)
        for seg in range(test_X.shape[3]):
            t = np.arange(0, test_X.shape[2], 1)
            plt.plot(t, test_X[0, channel, :], color='b', linewidth=0.2)
        plt.setp(ax1.get_xticklabels(), fontsize=2)
        plt.xlim(1, test_X.shape[2])
        plt.show()
    current_path = given_path
    plt.savefig(current_path)


def main():
    """Generate samples"""

    # Training experiment run UUID
    # 这个是没做normalized 成功案例
    run_uuid = "7b3acd44382a11edb808e4434b7714aa"
    # 这个是没做normalized 成功案例
    # run_uuid = "def293fe38dd11ed949ae4434b7708ee"

    # Start an evaluation
    experiment.evaluate()

    # Create configs
    configs = Configs()
    # Load custom configuration of the training run
    configs_dict = experiment.load_configs(run_uuid)
    # Set configurations
    experiment.configs(configs, configs_dict)

    # Initialize
    configs.init()

    # Set PyTorch modules for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model, 'subject_theta': configs.sub_theta})

    # Load training experiment
    experiment.load(run_uuid,8052)
    # window_size = 400
    # stack_size = 8
    configs.window_size = 224
    configs.stack_size = 8
    
    step_size = (750 - configs.window_size) // (configs.stack_size - 1)

    print(step_size)
    # Create sampler
    sampler = Sampler(diffusion=configs.diffusion,
                      image_channels=configs.image_channels,
                      device=configs.device, window_size=configs.window_size, step=step_size, stack_size=configs.stack_size)

    # Start evaluation
    with experiment.start():
        # No gradients
        with torch.no_grad():
            # Sample an image with an denoising animation
            sampler.sample_animation()

            if False:
                # Get some images fro data
                data = next(iter(configs.data_loader)).to(configs.device)

                # Create an interpolation animation
                sampler.interpolate_animate(data[0], data[1])


#
if __name__ == '__main__':
    main()
