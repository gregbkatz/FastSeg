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

class _Dataset(Dataset):
    _xs = []
    _ys = []

    def __init__(self, x_image_paths, label_image_paths, transform_x, transform_y):

        self.transform_x = transform_x
        self.transform_y = transform_y
        self.binary_classification = binary_classification
        self._xs = color_image_paths
        self._ys = label_image_paths
        self.normalize = transforms.Normalize((0), (1.0))

    def __getitem__(self, index):

        x = np.load(self._xs[index])
        img_x = x
        img_y = Image.open(self._ys[index])
        img_y = np.asarray(img_y)

        ## @Greg Need to add operations here

        return img_x, img_y

    def __len__(self):
        return len(self._xs)


class U_net(nn.Module):

    def __init__(self, in_channel, num_classes=3, start_filters=64, num_batchnorm_layers=0, dropout=0.1):

        super().__init__()
        self.bias = True

        self.conv1a = nn.Conv2d(in_channel, start_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1b = nn.Conv2d(start_filters, start_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1c = nn.Conv2d(2 * start_filters, start_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv1d = nn.Conv2d(start_filters, start_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.conv2a = nn.Conv2d(start_filters, 2 * start_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2b = nn.Conv2d(2 * start_filters, 2 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv2c = nn.Conv2d(4 * start_filters, 2 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv2d = nn.Conv2d(2 * start_filters, 2 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)

        self.conv3a = nn.Conv2d(2 * start_filters, 4 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv3b = nn.Conv2d(4 * start_filters, 4 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv3c = nn.Conv2d(8 * start_filters, 4 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv3d = nn.Conv2d(4 * start_filters, 4 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)

        self.conv4a = nn.Conv2d(4 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv4b = nn.Conv2d(8 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv4c = nn.Conv2d(16 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv4d = nn.Conv2d(8 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)

        self.conv5a = nn.Conv2d(8 * start_filters, 16 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)
        self.conv5b = nn.Conv2d(16 * start_filters, 16 * start_filters, kernel_size=3, stride=1, padding=1,
                                bias=self.bias)

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

        nn.init.kaiming_normal_(self.deconv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.deconv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.deconv4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.deconv5.weight, nonlinearity='relu')

        nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')

    def forward(self, x):

        # First layer
        x = self.dropout(x)

        # First layer
        a1a = self.relu(self.conv1a(x))
        a1b = self.relu(self.conv1b(a1a))

        p1 = self.pool(a1b)

        # Second layer
        a2a = self.relu(self.conv2a(p1))
        a2b = self.relu(self.conv2b(a2a))
        p2 = self.pool(a2b)

        # Third layer
        a3a = self.relu(self.conv3a(p2))
        a3a = self.dropout(a3a)
        a3b = self.relu(self.conv3b(a3a))
        p3 = self.pool(a3b)

        # Fourth layer
        a4a = self.relu(self.conv4a(p3))
        a4a = self.dropout2(a4a)
        a4b = self.relu(self.conv4b(a4a))
        p4 = self.pool(a4b)

        # Fifth layer
        a5a = self.relu(self.conv5a(p4))
        a5a = self.dropout2(a5a)
        a5b = self.relu(self.conv5b(a5a))

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
        a1c = self.relu(self.conv1c(conc_1))
        a1d = self.relu(self.conv1d(a1c))

        # Convolves to (N,num_classes,H,W)
        scores = self.relu(self.conv6(a1d))

        return scores

def get_val_loss(model, loss_weights, val_loader):

    epoch_loss_val = 0.0

    with torch.no_grad():
        model.eval()

        for iteration, batch_sampled in enumerate(val_loader):

            x = batch_sampled[0]
            y = batch_sampled[1]

            x = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False, device=device)

            scores = model(x)
            loss = torch.nn.functional.cross_entropy(scores, y, weight=loss_weights)
            epoch_loss_val += float(loss)

    model.train()

    return epoch_loss_val

def train_model(model, optimizer, train_loader, loss_weights, val_loader, model_id, epochs):


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    print("num epochs to be trained", epochs)
    
    current_path = '/home/fast_seg/FastSeg/model_checkpoints/'
    save_path = current_path + 'dummy' + '.pt'  #@Greg Add path here" '.pt'
    torch.save(model, save_path)

    for e in range(epochs):

        epoch_loss_train = 0.0
        t1 = time.time()

        print("")
        print("Training epoch", e)

        for iteration, batch_sampled in enumerate(train_loader):

            x = batch_sampled[0]
            y = batch_sampled[1]

            x = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False, device=device)

            scores = model(x)
            loss = torch.nn.functional.cross_entropy(scores, y, weight=loss_weights)
            epoch_loss_train += float(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 1 == 0:
                print('Iteration, loss, time: ', str(iteration), float(loss), time.time() - t1)

        print("Epoch,  loss train:", e, float(epoch_loss_train))
        print()

        print('Saving model')

        save_path = current_path + str(model_id) + "-" + str(e) + '.pt'  #@Greg Add path here" '.pt'
        torch.save(model, save_path)

        if e % 5 == 0:
            print("Calculating validation loss:")
            val_loss = get_val_loss(model, loss_weights, val_loader)
            print(float(val_loss))
            scheduler.step(val_loss)

def main():
    minibatch_size = 16
    num_classes = 91
    resolution = (64,64)

    #input_folder_path_train =
    #label_folder_path_train = 
    #input_folder_path_val = 
    #label_folder_path_val = 

    transform = transforms.Compose([CocoDataset.Rescale(resolution),
                               CocoDataset.ToTensor(), 
                               CocoDataset.Normalize()])

    seed = 1
    class_weights = [1.0] * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float, requires_grad=False, device=device)
    learning_rate = 1e-5
    bn = 0
    dp = 0.1
    f = 16
    wd = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dset_train = CocoDataset.CocoDataset('train2017', transform=transform, length=None)
    dset_val = CocoDataset.CocoDataset('val2017', transform=transform, length=None)

    train_loader = DataLoader(dset_train, batch_size=minibatch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)

    model_id = time.time()

    model = U_net(in_channel=3, num_classes=num_classes, start_filters=f, num_batchnorm_layers=bn, dropout=dp)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    train_model(model, optimizer, train_loader, class_weights, val_loader, model_id, epochs=20)

if __name__ == '__main__':
    main()
