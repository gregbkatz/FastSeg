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

def init_weights(m):
    nn.init.kaiming_normal_(m, nonlinearity='relu')

class UNetSep3(nn.Module):

    def sep_conv(self, in_f, out_f, layer):
        reg = nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1, bias=self.bias)
        width = nn.Conv2d(in_f, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        depth = nn.Conv2d(1, out_f, kernel_size=1, stride=1, padding=0, bias=self.bias)
       
        reg = nn.Sequential(reg, self.module_list.relu) 
        sep = nn.Sequential(width, self.module_list.relu, depth, self.module_list.relu)
        self.module_list.add_module('conv' + layer, reg)
        self.module_list.add_module('conv' + layer + '_sep', sep)

        return reg, sep

    def deconv(self, in_f, out_f, layer):
        deconv = nn.ConvTranspose2d(in_f, out_f, 2, stride=2, padding=0,
                                    bias=self.bias)
        #self.module_list.add_module('deconv' + layer, deconv)
        self.deconv_layers.append(deconv)
        return deconv

    def conv_layer(self, i, in_channel, start_filters):
        if i == 0:
            reg_a, sep_a = self.sep_conv(in_channel, start_filters, '1a')
        else: 
            reg_a, sep_a = self.sep_conv(2**(i-1)*start_filters, 2**i*start_filters, str(i+1) + 'a') 

        reg_b, sep_b = self.sep_conv(2**i*start_filters, 2**i*start_filters, str(i+1) + 'b') 
        reg_c, sep_c = self.sep_conv(2**(i+1)*start_filters, 2**i*start_filters, str(i+1) + 'c') 
        reg_d, sep_d = self.sep_conv(2**i*start_filters, 2**i*start_filters, str(i+1) + 'd') 

        if i == 0:
            encode = nn.Sequential(sep_a, sep_b)
        else: 
            encode = nn.Sequential(self.module_list.pool, sep_a, sep_b) 

        decode = nn.Sequential(sep_c, sep_d)
 
        if i > 0:
            deconv = self.deconv(2**(i)*start_filters, 2**(i-1)*start_filters, str(i+1))

        #self.module_list.add_module('encode' + str(i+1), encode)  
        #self.module_list.add_module('decode' + str(i+1), decode)  
        self.encode_layers.append(encode)
        self.decode_layers.append(decode)


    def __init__(self, in_channel, num_classes=3, start_filters=64, num_batchnorm_layers=0, dropout=0.1):

        super().__init__()
        self.bias = True

        self.module_list = nn.ModuleList()
        self.module_list.add_module('relu', nn.ReLU(inplace=True))
        self.module_list.add_module('pool', nn.MaxPool2d(2, stride=2))

        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        nlayers = 5
        for i in range(nlayers):
            self.conv_layer(i, in_channel, start_filters)
        
        self.module_list.add_module('conv_out', nn.Conv2d(start_filters, num_classes, kernel_size=1, stride=1, padding=0))

        #self.apply(init_weights)

    def forward(self, x):

        # Encode 
        a = []
        a.append(x)
        for i in range(len(self.encode_layers)):
            a.append(self.encode_layers[i](a[i]))

        # Decode
        d = a[-1]
        for i in range(len(self.deconv_layers), 0, -1):
            conc = torch.cat((a[i], self.deconv_layers[i-1](d)), 1)
            d = self.decode_layers[i-1](conc)

        # Convolves to (N,num_classes,H,W)
        scores = self.module_list.relu(self.module_list.conv_out(d))

        return scores

