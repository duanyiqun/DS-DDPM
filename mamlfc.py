# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Meta Learner """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from EEGNet import EEG_Net


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.parameter.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.parameter.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.softmax(F.linear(input_x, fc1_w, fc1_b), dim=1)
        return net

    def parameters(self):
        return self.vars


class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='pre', num_cls=4, embedding_size=4*2*25):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 4*2*25
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            self.encoder = EEG_Net()
        else:
            self.encoder = EEG_Net(mtl=False)
            self.pre_fc = nn.Sequential(nn.Linear(embedding_size, num_cls))

    def forward(self, inp):
        if self.mode=='pre' or self.mode=='origval':
            return self.pretrain_forward(inp)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        return F.softmax(self.pre_fc(self.encoder(inp)), dim=1)
