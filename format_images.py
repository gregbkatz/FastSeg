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
import pdb


np.set_printoptions(threshold=np.nan)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)


def main():
    resolution = (64,64)
    seed = 1

    transform = transforms.Compose([CocoDataset.Rescale(resolution),
                               CocoDataset.ToTensor(), 
                               CocoDataset.Normalize()])

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dset_train = CocoDataset.CocoDataset('train2017', transform=transform, length=None)
    dset_val = CocoDataset.CocoDataset('val2017', transform=transform, length=None)

    train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)

    drop_index = 3

    save_path_base = '/home/fast_seg/coco_pt2/'
    
    for iteration, batch_sampled in enumerate(train_loader):
        x = batch_sampled[0].cuda()
        y = batch_sampled[1].cuda()

        x = x.squeeze()
        y = y.squeeze()

        if(iteration % drop_index == 0):
            print("Saving iteration", iteration)
            x_name = str(iteration/3).zfill(7) + '_x.pt'
            y_name = str(iteration/3).zfill(7) + '_y.pt'
            torch.save(x, save_path_base + x_name)
            torch.save(y, save_path_base + y_name)

    print("Finished")

if __name__ == '__main__':
    main()
