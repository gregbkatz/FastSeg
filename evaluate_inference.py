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
import UNet
import argparse
import pdb
import U_Net_separable
import UNetSep2
import train_model

np.set_printoptions(threshold=np.nan, suppress=True, precision=4)

def evaluate(model, dset, device):

    evaluation_time = 0
    with torch.no_grad():
        model.eval()
        for i in range(len(dset)):
            t0 = time.time()
            x = dset[i][0]
            y = dset[i][1]
            x = x[None,:,:,:]
            y = y[None,:,:]

            x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device=device)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False, device=device)

            t1 = time.time()
            scores = model(x)
            t2 = time.time()
  
            # skip first iteration for warm up 
            if i > 0:
                #print("data load time: {:6.4f}, model evaluate time: {:6.4f}".format(t1 - t0, t2 -t1))
                evaluation_time += time.time() - t1

    return evaluation_time / (len(dset) - 1)

def main(args):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    device = torch.device(args.d)
    print("Using device", device)

    print("Initializing data loader")
    val_dir = "/home/fast_seg/coco_pt/val"
    dset = train_model.coco_custom_Dataset(val_dir, length=args.n)

    print("Loading model")
    if args.d == "cpu":
        model = torch.load(args.model_checkpoint, map_location=lambda storage, loc: storage)
    else: 
        model = torch.load(args.model_checkpoint)

    print("Evaluating inference time")
    avg_time = evaluate(model, dset, device)
    print("avg eval time: {:6.4f} rate: {:6.4f} Hz".format(avg_time, 1/avg_time))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint', type=str,
                        help='path to model checkpoint')
    parser.add_argument('-n', type=int, default = 100,
                        help='# of images to evaluate')
    parser.add_argument('-d', type=str, default = "cuda:0",
                        help='device, should be cuda:0 or cpu')
    args = parser.parse_args()
    main(args)
