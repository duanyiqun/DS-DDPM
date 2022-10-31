# DS-DDPM
Implementation of Domain Specific Denoising Diffusion Probabilistic Models for Brain Dynamics/EEG Signals

The intuition is simple, traditional EEG denoising methods normal apply blind source separation or frequency filtering.
We apply domain generation to model human subjects difference separation as a summation of domain generation and clean 
content denoising as figure shows below.

<img src=./imgs/mathmodel.png width = "780" height = "200" alt="图片名称" align=center />


Note: Thanks for implementation from labml as our reference
```https://nn.labml.ai/diffusion/ddpm/index.html```

Please install required pip packages
```ssh
pip install tensorboardx
pip install labml
pip install scikit-learn
```

The training entry scripts:
```bash
python comon_two_stream_sep_gaussion_organal_arc_margin_nomalizedy.py
```

The generated example of EEG signals with separated noise (green)

<img src=./visualization/noise_995_subject_%5B1%5D.png width = "500" height = "300" alt="图片名称" align=center />

This method may also be used for other time sequence signals. 
