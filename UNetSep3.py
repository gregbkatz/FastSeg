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

    # Returns both regular and separable version of convolution
    # Creates Sequential modules that include relu    
    # Includes kaiming initialization
    # reg = relu(Conv2d(x))
    # sep = relu(Conv2d(relue(Conv2d(x))))
    def sep_conv(self, in_f, out_f):
        reg = nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1, bias=self.bias)
        width = nn.Conv2d(in_f, in_f, kernel_size=3, stride=1, padding=1, bias=self.bias, groups=in_f)
        depth = nn.Conv2d(in_f, out_f, kernel_size=1, stride=1, padding=0, bias=self.bias)
       
        nn.init.kaiming_normal_(reg.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(width.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(depth.weight, nonlinearity='relu')
 
        reg = nn.Sequential(reg, self.module_list.relu) 
        sep = nn.Sequential(width, self.module_list.relu, depth, self.module_list.relu)

        return reg, sep

    # Returns a deconv module with initialization
    def deconv(self, in_f, out_f):
        deconv = nn.ConvTranspose2d(in_f, out_f, 2, stride=2, padding=0, bias=self.bias)

        nn.init.kaiming_normal_(deconv.weight, nonlinearity='relu')

        return deconv

    # Creates encode, deconv, and decode modules for a layer
    def add_layer(self, i, in_channel, start_filters, do_dropout):
        # a and b are for encoding
        if i == 0:
            reg_a, sep_a = self.sep_conv(in_channel, start_filters)
        else: 
            reg_a, sep_a = self.sep_conv(2**(i-1)*start_filters, 2**i*start_filters) 

        reg_b, sep_b = self.sep_conv(2**i*start_filters, 2**i*start_filters) 

        # c and de are for decoding
        reg_c, sep_c = self.sep_conv(2**(i+1)*start_filters, 2**i*start_filters) 
        reg_d, sep_d = self.sep_conv(2**i*start_filters, 2**i*start_filters) 
   
        if self.do_sep:
            conva, convb, convc, convd = sep_a, sep_b, sep_c, sep_d
        else:
            conva, convb, convc, convd = reg_a, reg_b, reg_c, reg_d

        # Add dropout between a and b if requested 
        if do_dropout:
            encode = nn.Sequential(conva, self.module_list.dropout, convb)
        else:
            encode = nn.Sequential(conva, convb)
    
        # Including max pooling at beginning of encoding layers with exception
        # of the first layer
        if i > 0:
            encode = nn.Sequential(self.module_list.pool, encode)

        decode = nn.Sequential(convc, convd)
 
        # There are one fewer deconv layers so skip it on the first layer
        deconv = None
        if i > 0:
            deconv = self.deconv(2**(i)*start_filters, 2**(i-1)*start_filters)

        return encode, deconv, decode

    def __init__(self, in_channel, num_classes=3, start_filters=64, dropout=0.1, nlayers=5, do_sep=1):

        super().__init__()
        self.bias = True
        self.do_sep = do_sep

        self.module_list = nn.ModuleList()
        self.module_list.add_module('relu', nn.ReLU(inplace=True))
        self.module_list.add_module('pool', nn.MaxPool2d(2, stride=2))
        self.module_list.add_module('dropout', nn.Dropout(p=dropout))

        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        for i in range(nlayers):
            do_dropout = i > nlayers-3
            encode, deconv, decode, = self.add_layer(i, in_channel, start_filters, do_dropout)
            self.encode_layers.append(encode)
            self.decode_layers.append(decode)
            if deconv: 
                self.deconv_layers.append(deconv)


        # Add one more convlution for final output        
        conv_out = nn.Conv2d(start_filters, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(conv_out.weight, nonlinearity='relu')
        self.module_list.add_module('conv_out', conv_out)

    def forward(self, x):

        # Encode 
        # Cache activations to pass into decoder
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

