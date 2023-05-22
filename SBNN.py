import os
import torch
from torch import nn, Module
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from newConv import Conv2d


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Peter is using {} device'.format(device))
print('-----------------------------------------------------')


class SBNN(nn.Module):
    def __init__(self):
        super(SBNN, self).__init__()
        self.conv1 = Conv2d(kernel_size=3, stride=1, padding='SAME', )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = SBNN().to(device)
print(model)
