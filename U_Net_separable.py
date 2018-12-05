import glob
import os
import timeit
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import colors as mcolors
from PIL import Image

import utils
import pdb

class U_net_separable(nn.Module):

    def __init__(self, in_channel, num_classes=3, start_filters=64, num_batchnorm_layers=0, dropout=0.1):

        super().__init__()
        self.bias = True

        self.conv1a_1 = nn.Conv2d(in_channel, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1a = nn.Conv2d(1, start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv1b_1 = nn.Conv2d(start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1b = nn.Conv2d(1, start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv1c_1 = nn.Conv2d(2 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1c = nn.Conv2d(1, start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv1d_1 = nn.Conv2d(start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1d = nn.Conv2d(1, start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        ####

        self.conv2a_1 = nn.Conv2d(start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2a = nn.Conv2d(1, 2 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv2b_1 = nn.Conv2d(2 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2b = nn.Conv2d(1, 2 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv2c_1 = nn.Conv2d(4 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2c = nn.Conv2d(1, 2 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv2d_1 = nn.Conv2d(2 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2d = nn.Conv2d(1, 2 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        ###

        self.conv3a_1 = nn.Conv2d(2 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv3a = nn.Conv2d(1, 4 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv3b_1 = nn.Conv2d(4 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv3b = nn.Conv2d(1, 4 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv3c_1 = nn.Conv2d(8 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv3c = nn.Conv2d(1, 4 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv3d_1 = nn.Conv2d(4 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv3d = nn.Conv2d(1, 4 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        ###


        self.conv4a_1 = nn.Conv2d(4 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv4a = nn.Conv2d(1, 8 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv4b_1 = nn.Conv2d(8 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv4b = nn.Conv2d(1, 8 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)


        self.conv4c_1 = nn.Conv2d(16 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv4c = nn.Conv2d(1, 8 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)


        self.conv4d_1 = nn.Conv2d(8 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv4d = nn.Conv2d(1, 8 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        ###

        self.conv5a_1 = nn.Conv2d(8 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv5a = nn.Conv2d(1, 16 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv5b_1 = nn.Conv2d(16 * start_filters, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv5b = nn.Conv2d(1, 16 * start_filters, kernel_size=1, stride=1, padding=0, bias=self.bias)

        self.conv6 = nn.Conv2d(start_filters, num_classes, kernel_size=1, stride=1, padding=0)

        self.deconv2 = nn.ConvTranspose2d(2 * start_filters, start_filters, 2, stride=2, padding=0,
                                          bias=self.bias)
        self.deconv3 = nn.ConvTranspose2d(4 * start_filters, 2 * start_filters, 2, stride=2, padding=0,
                                          bias=self.bias)
        self.deconv4 = nn.ConvTranspose2d(8 * start_filters, 4 * start_filters, 2, stride=2, padding=0,
                                          bias=self.bias)

        self.deconv5 = nn.ConvTranspose2d(16 * start_filters, 8 * start_filters, 2, stride=2, padding=0, bias=self.bias)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.num_batchnorm_layers = num_batchnorm_layers

        self.bns = nn.BatchNorm2d(start_filters)
        self.bn2s = nn.BatchNorm2d(2 * start_filters)
        self.bn4s = nn.BatchNorm2d(4 * start_filters)
        self.bn8s = nn.BatchNorm2d(8 * start_filters)
        self.bn16s = nn.BatchNorm2d(16 * start_filters)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)

        nn.init.kaiming_normal_(self.conv1a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5b.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1c.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2c.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3c.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4c.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3d.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4d.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a_1.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1b_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2b_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3b_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4b_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5b_1.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1c_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2c_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3c_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4c_1.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1d_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2d_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3d_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4d_1.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.deconv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.deconv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.deconv4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.deconv5.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')

    def forward(self, x):

        # First layer
        #x = self.dropout(x)

        # First layer
        a1a_1 = self.relu(self.conv1a_1(x))
        a1a = self.relu(self.conv1a(a1a_1))

        a1b_1 = self.relu(self.conv1b_1(a1a))
        a1b = self.relu(self.conv1b(a1b_1))

        p1 = self.pool(a1b)
        p1 = self.bns(p1)

        # Second layer
        a2a_1 = self.relu(self.conv2a_1(p1))
        a2a = self.relu(self.conv2a(a2a_1))

        a2b_1 = self.relu(self.conv2b_1(a2a))
        a2b = self.relu(self.conv2b(a2b_1))

        p2 = self.pool(a2b)
        p2 = self.bn2s(p2)

        # Third layer
        a3a_1 = self.relu(self.conv3a_1(p2))
        a3a = self.relu(self.conv3a(a3a_1))

        a3a = self.dropout(a3a)

        a3b_1 = self.relu(self.conv3b_1(a3a))
        a3b = self.relu(self.conv3b(a3b_1))

        p3 = self.pool(a3b)
        p3 = self.bn4s(p3)

        # Fourth layer
        a4a_1 = self.relu(self.conv4a_1(p3))
        a4a = self.relu(self.conv4a(a4a_1))

        a4a = self.dropout(a4a)

        a4b_1 = self.relu(self.conv4b_1(a4a))
        a4b = self.relu(self.conv4b(a4b_1))

        p4 = self.pool(a4b)
        p4 = self.bn8s(p4)

        # Fifth layer
        a5a_1 = self.relu(self.conv5a_1(p4))
        a5a = self.relu(self.conv5a(a5a_1))
        a5a = self.dropout(a5a)
        a5b_1 = self.relu(self.conv5b_1(a5a))
        a5b = self.relu(self.conv5b(a5b_1))

        # deconv layers
        conc_4 = torch.cat((a4b, self.deconv5(a5b)), 1)
        a4c_1 = self.relu(self.conv4c_1(conc_4))
        #pdb.set_trace()
        a4c = self.relu(self.conv4c(a4c_1))
        a4c = self.dropout(a4c)
        a4d_1 = self.relu(self.conv4d_1(a4c))
        a4d = self.relu(self.conv4d(a4d_1))

        conc_3 = torch.cat((a3b, self.deconv4(a4d)), 1)

        a3c_1 = self.relu(self.conv3c_1(conc_3))
        a3c = self.relu(self.conv3c(a3c_1))

        a3d_1 = self.relu(self.conv3d_1(a3c))
        a3d = self.relu(self.conv3d(a3d_1))

        conc_2 = torch.cat((a2b, self.deconv3(a3d)), 1)

        a2c_1 = self.relu(self.conv2c_1(conc_2))
        a2c = self.relu(self.conv2c(a2c_1))

        a2d_1 = self.relu(self.conv2d_1(a2c))
        a2d = self.relu(self.conv2d(a2d_1))

        conc_1 = torch.cat((a1b, self.deconv2(a2d)), 1)

        a1c_1 = self.relu(self.conv1c_1(conc_1))
        a1c = self.relu(self.conv1c(a1c_1))

        a1d_1 = self.relu(self.conv1d_1(a1c))
        a1d = self.relu(self.conv1d(a1d_1))

        # Convolves to (N,num_classes,H,W)
        scores = self.conv6(a1d)

        return scores
