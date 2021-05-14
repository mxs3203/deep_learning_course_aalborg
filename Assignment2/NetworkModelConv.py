

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class ConvNet(torch.nn.Module):

    def __init__(self,conv1_size,conv2_size, pool_size, kernel_size, stride=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_size, kernel_size=(kernel_size, kernel_size), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=conv1_size, out_channels=conv2_size, kernel_size=(kernel_size, kernel_size), stride=(stride,stride))
        self.pool1 = nn.MaxPool2d(pool_size, pool_size)

        self.relu = nn.ReLU()
        flatten_by_width = self.flatten_size(40, kernel_size, pool_size, stride)
        flatten_by_height = self.flatten_size(101, kernel_size, pool_size, stride)
        self.flatten_size = conv2_size*flatten_by_width*flatten_by_height
        print(self.flatten_size)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 3)

    def flatten_size(self, width_or_height, kernel_size, pool_size, stride):
        # first layer
        a = width_or_height - kernel_size + stride
        a = int(((a - pool_size)/pool_size) + stride)
        # second layer
        a = a - kernel_size + stride
        a = int(((a - pool_size)/pool_size) + stride)
        return a


    def forward(self, x):
        # conv -> relu -> max pool
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool1(self.relu(self.conv2(x)))
        # ff network
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x