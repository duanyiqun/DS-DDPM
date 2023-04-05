# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modified from: https://github.com/aliasvishnu/EEGNet
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Feature Extractor """
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class EEG_Net(nn.Module):

    def __init__(self, mtl=True):
        super(EEG_Net, self).__init__()
        self.Conv2d = nn.Conv2d
        self.conv1 = nn.Conv2d(7, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        x = x.contiguous().view(-1, 4 * 2 * 25)

        return x


class EEG_Net_8_Stack(nn.Module):

    def __init__(self, mtl=True):
        super(EEG_Net_8_Stack, self).__init__()
        self.Conv2d = nn.Conv2d
        self.conv1 = nn.Conv2d(8, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        # print(x.shape)
        x = x.contiguous().view(-1, 4 * 2 * 14)

        return x


class EEG_Net_1x(nn.Module):

    def __init__(self, mtl=True):
        super(EEG_Net_1x, self).__init__()
        self.Conv2d = nn.Conv2d
        self.conv1 = nn.Conv2d(7, 16, (1, 4), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(19, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

    def forward(self, x, verbose_debug=False):
        if verbose_debug:
            print('the original input shape is {}'.format(x.size()))
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        if verbose_debug:
            print('first conv layer with size (7, 16, (1, 22), the output shape is {}'.format(x.size()))
        x = x.permute(0, 3, 1, 2)
        # Layer 2
        x = self.padding1(x)
        if verbose_debug:
            print('first conv layer after first padding, the output shape is {}'.format(x.size()))
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        if verbose_debug:
            print('second conv layer with size (1, 4, (2, 32)), the output shape is {}'.format(x.size()))
        x = self.pooling2(x)
        if verbose_debug:
            print('second conv layer after second pooling, the output shape is {}'.format(x.size()))
        # Layer 3
        x = self.padding2(x)
        if verbose_debug:
            print('second conv layer after second padding, the output shape is {}'.format(x.size()))
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        if verbose_debug:
            print('third conv layer after nn.Conv2d(4, 4, (8, 4)), the output shape is {}'.format(x.size()))
        x = self.pooling3(x)
        x = x.contiguous().view(-1, 4 * 2 * 25)

        return x



if __name__ == "__main__":
    model = EEG_Net_8_Stack()
    sample_data = torch.randn(16, 8, 224, 22)
    # sample_out = model(sample_data, verbose_debug=True)
    sample_out = model(sample_data)
    print(sample_out.size())
