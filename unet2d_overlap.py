"""
---
title: A minimum implementation for DS_DDPM for start over with code comments
---
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
from src.EEGNet import EEG_Net_8_Stack
from src.unet_eeg_subject_emb import sub_gaussion
from src.unet_eeg import UNet
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from src.utils import gather
import numpy as np
import scipy.io as sio
import csv
import math

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

    def __init__(self, in_features, out_features, load_backbone = './assets/max_acc.pth', s=30.0, m=0.50, easy_margin=False):
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

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, sub_theta: nn.Module, sub_arc_head: nn.Module, debug=False, time_diff_constraint=True):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        self.sub_theta = sub_theta
        self.sub_arc_head = sub_arc_head
        self.time_diff_constraint = time_diff_constraint

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
                                    noise_content_kl_co = 1, arc_subject_co = 0.1, orgth_co = 2):
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
            print("the shape of subject_theta")
            print(subject_theta.shape)
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
        organal_squad = torch.bmm(eps_theta.view(eps_theta.shape[0]*eps_theta.shape[1], eps_theta.shape[2],eps_theta.shape[3]), subject_theta.view(subject_theta.shape[0]*subject_theta.shape[1], subject_theta.shape[2], subject_theta.shape[3]).permute(0,2,1))
        if debug:
            print("logging with organal_squad")
            print(organal_squad.shape)
        ones = torch.ones(eps_theta.shape[0]*eps_theta.shape[1], eps_theta.shape[2], eps_theta.shape[2], dtype=torch.float32, device='cuda') # (N * C) * H * H   
        diag = torch.eye(eps_theta.shape[2], dtype=torch.float32,device='cuda') # (N * C) * H * H
        loss_orth = ((organal_squad * (ones - diag).to('cuda')) ** 2).mean()
        if debug:
            print("logging with loss_orth")
            print(loss_orth)
        subject_arc_logit = self.sub_arc_head(subject_theta.permute(0,3,2,1), s)
        subject_arc_loss = F.cross_entropy(subject_arc_logit, s.long())
        # return F.mse_loss(noise, eps_theta + subject_theta) + 0.01 * 1/noise_conent_kl, constraint_panelty, noise_conent_kl
        # return F.mse_loss(noise, eps_theta + subject_theta) - noise_content_kl_co * noise_conent_kl + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss
        if self.time_diff_constraint:
            return F.mse_loss(noise, eps_theta + subject_theta) + orgth_co * loss_orth + arc_subject_co *subject_arc_loss + 0.1 * constraint_panelty, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss, orgth_co * loss_orth
        else:
            return F.mse_loss(noise, eps_theta + subject_theta) + orgth_co * loss_orth + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss, orgth_co * loss_orth

class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    sub_theta: sub_gaussion
    sub_archead: ArcMarginHead
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    eeg_channels: int = 22
    # Image size
    # image_size: int = 32
    window_size: int = 224
    # window_size: int = 400
    stack_size: int = 8
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps $T$
    n_steps: int = 1_000
    # Batch size
    # batch_size: int = 64
    batch_size: int = 32
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5
    arc_in = 4*2*14
    arc_out = 9

    # Number of training epochs
    epochs: int = 1_000

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam
    optimizer_noise: torch.optim.Adam

    def init(self):
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        self.eps_model = UNet(
            eeg_channels=self.eeg_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.sub_theta = sub_gaussion(
            eeg_channels=self.eeg_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.sub_archead = ArcMarginHead(
            self.arc_in, self.arc_out, 
            load_backbone = '/content/DS-DDPM/assets/max_acc.pth',
            s=30.0, m=0.50, easy_margin=False
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
            sub_theta=self.sub_theta,
            sub_arc_head=self.sub_archead,
            debug=False,
        )

        

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam([
                        {'params': self.eps_model.parameters(), 'lr': self.learning_rate} ,
                        {'params': self.sub_theta.parameters(), 'lr': self.learning_rate },
                        {'params': self.sub_archead.arcpro.parameters(), 'lr': self.learning_rate } ], lr=self.learning_rate)
        # self.optimizer_noise = torch.optim.Adam(self.sub_theta.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        self.optimizer_noise = torch.optim.Adam(self.sub_theta.parameters(), lr=self.learning_rate)

        # Image logging
        # tracker.set_image("sample", True)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.eeg_channels, self.window_size, self.stack_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            # tracker.save('sample', x)

    def train(self):
        """
        ### Train
        """
        metric_file = '/content/DS-DDPM/metrics/none_constraint_subject_gaussion_unet_att_loss_ep100_withorga_arc_normalized.csv'
        loss_target = open(metric_file, 'a+')
        target_writer = csv.writer(loss_target)
        # target_writer.writerow(['loss', 'time_period_diff'])


        # Iterate through the dataset
        for data, label in monit.iterate('Train', self.data_loader):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            data = torch.permute(data, (0,3,2,1))
            data = data.to(self.device)
            label = label.float().to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            # loss = self.diffusion.loss(data)
            loss, time_period_diff, noise_conent_kl, sub_arc_loss, loss_orth = self.diffusion.loss_with_diff_constraint(data, label)
            # return F.mse_loss(noise, eps_theta + subject_theta) + orgth_co * loss_orth + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss, orgth_co * loss_orth

            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            self.optimizer_noise.step()
            # Track the loss
            # tracker.save('loss', loss - time_period_diff)
            tracker.save('loss', loss)
            tracker.save('diffusion_loss', loss - loss_orth -sub_arc_loss)
            tracker.save('time_seg_diff', time_period_diff)
            tracker.save('noise_conent_kl', noise_conent_kl)
            tracker.save('sub_arc_loss', sub_arc_loss)
            tracker.save('loss_orth', loss_orth)
            target_writer.writerow([loss.detach().cpu().numpy(), (loss - loss_orth -sub_arc_loss).detach().cpu().numpy(), time_period_diff.detach().cpu().numpy(), noise_conent_kl.detach().cpu().numpy(), sub_arc_loss.detach().cpu().numpy(), loss_orth.detach().cpu().numpy()])
        loss_target.close()


    def run(self):
        """
        ### Training loop
        """
        for inicator in monit.loop(self.epochs):
            # Train the model
            print(inicator)
            self.train()
            # Sample some images
            self.sample()
            # New line in the console
            tracker.new_line()
            # Save the model
            if inicator % 30 == 0:
                experiment.save_checkpoint()


class DatasetLoader_BCI_IV_signle(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder=None, train_aug=False, subject_id=3):

        subject_id = subject_id
        if datafolder is None:
            data_folder = '../data'
        else:
            data_folder = datafolder
        data = sio.loadmat(data_folder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
        test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
        train_X = data["train_x"][:, :, 750:1500]

        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y -= 1
        test_y -= 1
        window_size = 224
        step = 75 # 这里必须保证产出的tensor 是偶数，这里是超大overlap的形式
        # window_size = 400
        # step = 50 # 这里必须保证产出的tensor 是偶数，这里是超大overlap的形式
        n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        print("The shape of sample x0 is {}".format(test_win_x.shape))

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label


class DatasetLoader_BCI_IV_mix(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder=None, train_aug=False):

        subject_id = 'all'
        if datafolder is None:
            data_folder = '../data'
        else:
            data_folder = datafolder
        data = sio.loadmat(data_folder + "/mix_sub/mix_subject_data_" + str(subject_id) + ".mat")
        test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
        train_X = data["train_x"][:, :, 750:1500]

        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y -= 1
        test_y -= 1
        # window_size = 400
        # step = 45
        window_size = 224
        step = 75
        n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        print('The shape of x is {}'.format(test_win_x.shape))

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label


class DatasetLoader_BCI_IV_mix_subjects(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder, train_aug=False):
        test_X_list = []
        train_X_list = []
        test_y_list = []
        train_y_list = []

        for i in range(9):
            subject_id = i + 1
            if datafolder is None:
                data_folder = '../data'
            else:
                data_folder = datafolder

            data = sio.loadmat(data_folder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
            test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
            train_X = data["train_x"][:, :, 750:1500]

            test_y = np.ones((test_X.shape[0],)) * i
            train_y = np.ones((train_X.shape[0],)) * i
            # print(test_y.shape)
            # print(train_y.shape)
            test_X_list.append(test_X)
            train_X_list.append(train_X)
            test_y_list.append(test_y)
            train_y_list.append(train_y)

        test_X = np.vstack(test_X_list)
        train_y = np.concatenate(train_y_list, axis=0)
        train_X = np.vstack(train_X_list)
        test_y = np.concatenate(test_y_list, axis=0)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X.shape)
        print(train_y.shape)
        # train_y -= 1
        # test_y -= 1
        window_size = 224
        step = 75
        n_channel = 22
        # n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ## comment out F.normalize if you want unconstrained diffusion (without normalize to 1)
        # data, label = F.normalize(torch.Tensor(self.data[i]), p=2, dim=2), self.label[i]
        data, label = torch.Tensor(self.data[i]), self.label[i]
        return data, label


@option(Configs.dataset, 'bci_comp_iv')
def datasetLoader_BCI_IV_signle(c: Configs):
    """
    Create BCI IV dataset
    """

    return DatasetLoader_BCI_IV_signle('train', datafolder='/content/DS-DDPM/DS-DDPM-data', subject_id=3)


@option(Configs.dataset, 'bci_comp_iv_full_mix')
def datasetLoader_BCI_IV_mix_subjects(c: Configs):
    """
    Create BCI IV dataset
    """
    return DatasetLoader_BCI_IV_mix_subjects('train', datafolder='/content/DS-DDPM/DS-DDPM-data')


def main():
    # Create experiment
    # experiment.create(name='unet_subjects_double_step_classifier_free_baseline', writers={'screen', 'comet'})
    # experiment.create(name='none_constraint_subject_gaussion_unet_att_loss_ep100_withorga_arc', writers={'screen', 'comet'})
    experiment.create(name='none_constraint_subject_gaussion_unet_att_loss_ep100_withorga_arc', writers={'screen', 'comet'})
    # experiment.create(name='unet2d_ds_unc', writers={'screen', 'comet', 'wandb'})


    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {
        # 'dataset': 'bci_comp_iv_full_mix',  # 'bci_comp_iv' bci_comp_iv_full_mix
        'dataset': 'bci_comp_iv_full_mix',
        'eeg_channels': 22,  # 1,
        'epochs': 100,  # 5,
    })

    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model, 'subject_theta': configs.sub_theta})

    # Start and run the training loop
    with experiment.start():
        configs.run()


#
if __name__ == '__main__':
    main()
 