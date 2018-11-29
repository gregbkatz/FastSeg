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


save_path_train = '/home/fast_seg/coco_pt/train/'
save_path_val = '/home/fast_seg/coco_pt/val/'
np.set_printoptions(threshold=np.nan)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)

def saveToFile(loader, save_path):
    for iteration, sample in enumerate(loader):
        base_name = save_path + str(iteration).zfill(7)
        print("Saving " + base_name)
        x = sample[0].squeeze()
        y = sample[1].squeeze()
        torch.save(x, base_name + "_x.pt")
        torch.save(y, base_name + "_y.pt")

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

    saveToFile(train_loader, save_path_train)
    saveToFile(val_loader, save_path_val)
    
    print("Finished")

if __name__ == '__main__':
    main()
