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

import CocoDataset
import utils
import pdb

NUM_CLASSES = 3

class UNetSep2(nn.Module):

    def sep_conv(self, in_f, out_f):
        reg = nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1, bias=self.bias)
        width = nn.Conv2d(in_f, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        depth = nn.Conv2d(1, out_f, kernel_size=1, stride=1, padding=0, bias=self.bias)
        return reg, width, depth


    def __init__(self, in_channel, num_classes=3, start_filters=64, num_batchnorm_layers=0, dropout=0.1):

        super().__init__()
        self.bias = True

        self.conv1a, self.conv1a_1, self.conv1a_2 = self.sep_conv(in_channel, start_filters)
        self.conv1b, self.conv1b_1, self.conv1b_2 = self.sep_conv(start_filters, start_filters)
        self.conv1c, self.conv1c_1, self.conv1c_2 = self.sep_conv(2 * start_filters, start_filters)
        self.conv1d, self.conv1d_1, self.conv1d_2 = self.sep_conv(start_filters, start_filters)

        self.conv2a, self.conv2a_1, self.conv2a_2 = self.sep_conv(start_filters, 2 * start_filters)
        self.conv2b, self.conv2b_1, self.conv2b_2 = self.sep_conv(2 * start_filters, 2 * start_filters)
        self.conv2c, self.conv2c_1, self.conv2c_2 = self.sep_conv(4 * start_filters, 2 * start_filters)
        self.conv2d, self.conv2d_1, self.conv2d_2 = self.sep_conv(2 * start_filters, 2 * start_filters)

        self.conv3a, self.conv3a_1, self.conv3a_2 = self.sep_conv(2 * start_filters, 4 * start_filters)
        self.conv3b, self.conv3b_1, self.conv3b_2 = self.sep_conv(4 * start_filters, 4 * start_filters)
        self.conv3c, self.conv3c_1, self.conv3c_2 = self.sep_conv(8 * start_filters, 4 * start_filters)
        self.conv3d, self.conv3d_1, self.conv3d_2 = self.sep_conv(4 * start_filters, 4 * start_filters)

        self.conv4a, self.conv4a_1, self.conv4a_2 = self.sep_conv(4 * start_filters, 8 * start_filters)
        self.conv4b, self.conv4b_1, self.conv4b_2 = self.sep_conv(8 * start_filters, 8 * start_filters)
        self.conv4c, self.conv4c_1, self.conv4c_2 = self.sep_conv(16 * start_filters, 8 * start_filters)
        self.conv4d, self.conv4d_1, self.conv4d_2 = self.sep_conv(8 * start_filters, 8 * start_filters)

        self.conv5a, self.conv5a_1, self.conv5a_2 = self.sep_conv(8 * start_filters, 16 * start_filters)
        self.conv5b, self.conv5b_1, self.conv5b_2 = self.sep_conv(16 * start_filters, 16 * start_filters)

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
        nn.init.kaiming_normal_(self.conv1a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1a_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2a_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a_2.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1b_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1b_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2b_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2b_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3a_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4a_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5b.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5a_2.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1c.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2c.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3c.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4c.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3d.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4d.weight, nonlinearity='relu')

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
        a1a = self.relu(self.conv1a_2(a1a_1))
        a1b_1 = self.relu(self.conv1b_1(a1a))
        a1b = self.relu(self.conv1b_2(a1b_1))

        p1 = self.pool(a1b)
        p1 = self.bns(p1)

        # Second layer
        a2a_1 = self.relu(self.conv2a_1(p1))
        a2a = self.relu(self.conv2a_2(a2a_1))
        a2b_1 = self.relu(self.conv2b_1(a2a))
        a2b = self.relu(self.conv2b_2(a2b_1))
        p2 = self.pool(a2b)
        p2 = self.bn2s(p2)

        # Third layer
        a3a_1 = self.relu(self.conv3a_1(p2))
        a3a = self.relu(self.conv3a_2(a3a_1))
        a3a = self.dropout(a3a)
        a3b_1 = self.relu(self.conv3b_1(a3a))
        a3b = self.relu(self.conv3b_2(a3b_1))
        p3 = self.pool(a3b)
        p3 = self.bn4s(p3)

        # Fourth layer
        a4a_1 = self.relu(self.conv4a_1(p3))
        a4a = self.relu(self.conv4a_2(a4a_1))
        a4a = self.dropout2(a4a)
        a4b_1 = self.relu(self.conv4b_1(a4a))
        a4b = self.relu(self.conv4b_2(a4b_1))
        p4 = self.pool(a4b)
        p4 = self.bn8s(p4)

        # Fifth layer
        a5a_1 = self.relu(self.conv5a_1(p4))
        a5a = self.relu(self.conv5a_2(a5a_1))
        a5a = self.dropout2(a5a)
        a5b_1 = self.relu(self.conv5b_1(a5a))
        a5b = self.relu(self.conv5b_2(a5b_1))

        # deconv layers
        conc_4 = torch.cat((a4b, self.deconv5(a5b)), 1)
        a4c = self.relu(self.conv4c(conc_4))
        a4c = self.dropout2(a4c)
        a4d = self.relu(self.conv4d(a4c))

        conc_3 = torch.cat((a3b, self.deconv4(a4b)), 1)
        a3c = self.relu(self.conv3c(conc_3))
        a3d = self.relu(self.conv3d(a3c))

        conc_2 = torch.cat((a2b, self.deconv3(a3d)), 1)
        a2c = self.relu(self.conv2c(conc_2))
        a2d = self.relu(self.conv2d(a2c))

        conc_1 = torch.cat((a1b, self.deconv2(a2d)), 1)
        #a1c_1 = self.relu(self.conv1c_1(conc_1))
        #a1c = self.relu(self.conv1c_2(a1c_1))
        a1c = self.relu(self.conv1c(conc_1))
        #a1d_1 = self.relu(self.conv1d_1(a1c))
        #a1d = self.relu(self.conv1d_2(a1d_1))
        a1d = self.relu(self.conv1d(a1c))

        # Convolves to (N,num_classes,H,W)
        scores = self.relu(self.conv6(a1d))

        return scores

