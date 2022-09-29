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
from diffusion_uncon_bci_compiv import Configs
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from utils import gather


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

events_id = {'1': 'Left', '2': 'Right', '3': 'Foot', '4': 'Tongue'}

class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

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

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
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

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_simple(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)


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

    def sample_animation(self, n_frames: int = 1000, create_video: bool = False, overlap_cover = True, debug=False):
        """
        #### Sample an image step-by-step using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        We sample an image step-by-step using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$ and at each step
        show the estimate
        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        """

        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([1, self.image_channels, self.window_size, self.stack_size], device=self.device)

        # Interval to log $\hat{x}_0$
        interval = self.n_steps // n_frames
        # Frames for video
        frames = []
        # Sample $T$ steps
        real_cureves_list = []
        for t_inv in monit.iterate('Denoise', self.n_steps):
            # $t$
            t_ = self.n_steps - t_inv - 1
            # $t$ in a tensor
            t = xt.new_full((1,), t_, dtype=torch.long)
            # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
            eps_theta = self.eps_model(xt, t)
            # print(eps_theta.size())
            real_waves_list = []

            if t_ % interval == 0:
                # Get $\hat{x}_0$ and add to frames
                x0 = self.p_x0(xt, t, eps_theta)
                
                # frames.append(x0[0])
                x0 = x0.detach().cpu().numpy()
                real_waves = np.ones((x0.shape[0], 22, 750))
                if overlap_cover:
                    for i in range(x0.shape[3]):
                        if debug:
                            print("{}-{}".format(0 + i* self.step, self.window_size + i * self.step))
                        real_waves[:,:,0 + i* self.step: self.window_size + i * self.step] = x0[:,:,:,i]
                else:
                    for i in range(x0.shape[3]):
                        if debug:
                            print("{}-{}".format(0 + i* self.step, self.window_size + i * self.step))
                        real_waves[:,:,0 + i* self.step: self.window_size + i * self.step] = x0[:,:,:,i]
                real_waves_list.append(real_waves)
                dataroot='/home/yiqduan/Data/ddpm/ddpm/visualization/bci_comp4/'
                curve_temp_path = "{}/curve_channel_diff{}.png".format(dataroot, "temp")
                # annimate_curve_change(test_X = real_waves[0:,0:6,:400], given_path=curve_temp_path)
                plt.clf()
                annimate_curve_change_with_diff(test_X = x0[0:,0:6,:,:], given_path=curve_temp_path)
                real_cureves_list.append(Image.open(curve_temp_path))
            
                # print(x0.shape)
                # if not create_video:
                    # self.show_image(x0[0], f"{t_}")
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt = self.p_sample(xt, t, eps_theta)
        current_path = "{}/curve_animate_id_channel_large_diff_{}.gif".format(dataroot, t_inv)
        # fig.colorbar(surf, shrink=0.2, aspect=5)
        gif.save(real_cureves_list, current_path)
                
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


def annimate_curve_change(test_X = None, test_y = None, given_path = None):
    plt.cla()
    if test_X is None:
        test_X = np.ones((1, 22,750))
    if test_y is None:
        test_y = np.ones(1,)
    print(test_X.shape)
    t = np.arange(0, test_X.shape[2], 1)
    for channel in range(test_X.shape[1]):
        ax1 = plt.subplot(test_X.shape[1], 1, channel+1)
        plt.plot(t, test_X[0, channel, :], color='b', linewidth=0.2)
        plt.setp(ax1.get_xticklabels(), fontsize=2)
        plt.xlim(1, test_X.shape[2])
        # plt.show()
    current_path = given_path
    plt.savefig(current_path)
        

def annimate_curve_change_with_diff(test_X = None, test_y = None, given_path = None, step=50):
    color_dict = {0: 'b', 1: 'r', 2:'g', 3:'y', 4:'c', 5:'m', 6:'lime', 7:'teal'}
    if test_X is None:
        test_X = np.ones((1, 22, 400, 8))
    if test_y is None:
        test_y = np.ones(1,)
    print(test_X.shape)
    # plt.figure(dpi=300,figsize=(18,12))
    for channel in range(test_X.shape[1]):
        ax1 = plt.subplot(test_X.shape[1], 1, channel+1)
        for seg in range(test_X.shape[3]):
            t = np.arange(0 + seg * step, test_X.shape[2] + seg * step, 1)
            plt.plot(t, test_X[0, channel, :, seg], color=color_dict[seg], linewidth=0.1)
        plt.setp(ax1.get_xticklabels(), fontsize=2)
        plt.xlim(1, test_X.shape[2] + (test_X.shape[3] -1)*step)
        # plt.show()
    current_path = given_path
    plt.savefig(current_path)



def main():
    """Generate samples"""

    # Training experiment run UUID
    run_uuid = "f28edb642c3c11ed949ae4434b7708ee"

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
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Load training experiment
    experiment.load(run_uuid,5369)
    # window_size = 400
    # stack_size = 8
    configs.window_size = 400
    configs.stack_size = 8
    
    step_size = (750 - configs.window_size) // (configs.stack_size - 1)

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
