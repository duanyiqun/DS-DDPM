"""
---
title: Utility functions for DDPM experiment
summary: >
  Utility functions for DDPM experiment
---

# Utility functions for [DDPM](index.html) experiemnt
"""
import torch.utils.data
import mne
import torch
import numpy as np
import matplotlib.pyplot as plt



def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def batch_construct_mne_info(batch_sample, batch_label, channel_names=None, channel_types=None, sfreq=250,
                             visual_len=5000, events_id=None):
    batch_shape = batch_sample.size()
    if len(batch_shape) == 4:
        batch_sample = batch_sample.view(-1, batch_shape[2], batch_shape[3])
        assert len(batch_sample.size()) == 3
        repeat_ratio = int(batch_sample.size()[0] / batch_shape[0])
        batch_label = torch.repeat_interleave(batch_label.unsqueeze(1), repeats=repeat_ratio, dim=1).view(-1, 1)
        print(batch_label.size(), batch_sample.shape)
    assert len(batch_sample.size()) == 3
    batch_sample = batch_sample.permute(0, 2, 1)
    if channel_names is None:
        channel_names = ['Fz', '2', '3', '4', '5', '6', '7', 'C3', '9', 'Cz', '11',
                        'C4', '13', '14', '15', '16', '17', '18', '19', 'Pz', '21', '22']
    if channel_types is None:
        channel_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                        'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    # print(len(channel_types))
    mne_info = mne.create_info(
        ch_names=channel_names,
        ch_types=channel_types,
        sfreq=sfreq
    )
    event_list = []
    batch_sample = batch_sample.numpy()
    batch_label = batch_label.numpy()
    for idx, label in enumerate(batch_label):
        if idx >= visual_len:
            break
        event_list.append([idx, 0, label[0]])
    events = np.array(event_list)
    if events_id is None:
        events_id = {'Left': 1, 'Right': 2, 'Foot': 3, 'Tongue': 4}
    tmin = -0.1
    custom_epochs = mne.EpochsArray(batch_sample[:visual_len, :, :], mne_info, events, tmin, events_id)
    return custom_epochs


def raw_construct_mne_info(raw_sample, raw_label, channel_names=None, channel_types=None, sfreq=250,
                             visual_len=5000, events_id=None):
    if channel_names is None:
        channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
                        'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    if channel_types is None:
        channel_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                        'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    # print(len(channel_types))
    mne_info = mne.create_info(
        ch_names=channel_names,
        ch_types=channel_types,
        sfreq=sfreq
    )
    event_list = []
    for idx, label in enumerate(raw_label):
        if idx >= visual_len:
            break
        event_list.append([idx, 0, int(label)])
    events = np.array(event_list)
    if events_id is None:
        events_id = {'Left': 1, 'Right': 2, 'Foot': 3, 'Tongue': 4}
    tmin = -0.1
    custom_epochs = mne.EpochsArray(raw_sample[:visual_len, :, :], mne_info, events, tmin, events_id)
    return custom_epochs


if __name__ == '__main__':
    sample_batch = torch.randn(16, 7, 400, 22)
    sample_labels = ((torch.rand(16, 1)) * 4 + 1).long()
    batch_construct_mne_info(sample_batch, sample_labels)


