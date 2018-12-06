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
import UNetSep3

import sys


#classes = utils.Classes('/home/fast_seg/coco/classes.txt')
#NUM_CLASSES = len(classes.classes)
NUM_CLASSES = 3
print("Num classes: ", NUM_CLASSES)
EPS = 0 #1e-8
train_dir = '/home/fast_seg/coco_pt/train/'
val_dir = '/home/fast_seg/coco_pt/val/'

np.set_printoptions(threshold=np.nan, suppress=True, precision=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_filenames(folder_path, pattern):
    return sorted(glob.glob(os.path.join(folder_path, pattern)))

class coco_custom_Dataset(Dataset):

    def __init__(self, path, length=None):
        self._xs = get_filenames(path, "*_x.pt")
        self._ys = get_filenames(path, "*_y.pt")
        self.length = len(self._xs)
        assert(self.length == len(self._ys))
        print("Found {} files in {}".format(self.length, path))
        if length is not None:
            print("Subsampling to {}".format(length))
            self.length = length

    def __getitem__(self, index):
        x = torch.load(self._xs[index])
        y = torch.load(self._ys[index])
        y[y>1] = 2
        assert(torch.max(y) <= 2)
        # y = y > 0
        return x, y

    def __len__(self):
        return self.length

def scores2preds(scores):
    _, preds = scores.max(1) # NxCxHxW -> NxHxW
    return preds

def get_val_loss(model, loss_weights, val_loader):

    loss = 0.0
    tp = np.zeros((NUM_CLASSES), dtype=int)
    fp = np.zeros((NUM_CLASSES), dtype=int)
    fn = np.zeros((NUM_CLASSES), dtype=int) 
    total = 0
    with torch.no_grad():
        model.eval()

        for iteration, batch_sampled in enumerate(val_loader):

            x = batch_sampled[0]
            y = batch_sampled[1]

            x = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False, device=device)

            scores = model(x)
            batch_loss = torch.nn.functional.cross_entropy(scores, y, weight=loss_weights)
            loss += float(batch_loss)

            total += y.shape[0] * y.shape[1] * y.shape[2]
            b_tp, b_fp, b_fn = get_metrics(scores, y)
            tp += b_tp
            fp += b_fp
            fn += b_fn

    model.train()

    # EPS not needed for iou because fp + fn + tp shoudl be > 0 
    iou = tp / (fp + fn + tp + EPS)
    precision = tp / (tp + fp + EPS) 
    recall = tp / (tp + fn + EPS)
    tn = total - tp - fp - fn
    accuracy = (tp + tn) / total
    total_accuracy = tp.sum() / total 
    print("Ground truth # examples")
    print(tp + fn)
    print("Prediction # examples")
    print(tp + fp)
    assert(total_accuracy >= 0)
    assert(total_accuracy <= 1)

    # Average loss over iterations
    loss = loss / (iteration + 1)
    return loss, iou, precision, recall, accuracy, total_accuracy

def get_metrics(scores, y):
    preds = scores2preds(scores)
    assert(preds.shape == y.shape)
    tp = np.zeros((NUM_CLASSES), dtype=int)
    fp = np.zeros((NUM_CLASSES), dtype=int)
    fn = np.zeros((NUM_CLASSES), dtype=int)
    for i in range(NUM_CLASSES):
        tp[i], fp[i], fn[i] = get_class_metrics(preds, y, i)

    return tp, fp, fn

def get_class_metrics(preds, y, class_id):
    preds = preds == class_id
    y = y == class_id
    tp = int(torch.sum(preds & y))
    fp = int(torch.sum(preds)) - tp
    fn = int(torch.sum(y)) - tp
    return tp, fp, fn

def train_model(model, optimizer, train_loader, loss_weights, val_loader, save_path, epochs, do_save=True):


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    print("num epochs to be trained", epochs)
    
    if do_save:
        torch.save(model, save_path + '/dummy.pt')

    for e in range(epochs):

        epoch_loss_train = 0.0
        t1 = time.time()

        print("")
        print("Training epoch", e)

        tp = np.zeros((NUM_CLASSES), dtype=int)
        fp = np.zeros((NUM_CLASSES), dtype=int)
        fn = np.zeros((NUM_CLASSES), dtype=int) 
        total = 0
        for i, batch_sampled in enumerate(train_loader):

            x = batch_sampled[0]
            y = batch_sampled[1]

            x = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False, device=device)

            scores = model(x)
            loss = torch.nn.functional.cross_entropy(scores, y, weight=loss_weights)
            epoch_loss_train += float(loss)

            total += y.shape[0] * y.shape[1] * y.shape[2]
            b_tp, b_fp, b_fn = get_metrics(scores, y)
            tp += b_tp
            fp += b_fp
            fn += b_fn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('Iteration: {:6} loss train: {:6.4f} time elapsed: {:6.1f}'.format(
                      i, epoch_loss_train/(i+1), time.time() - t1))

        train_iou = tp / (fp + fn + tp + EPS)
        train_precision = tp / (tp + fp + EPS) 
        train_recall = tp / (tp + fn + EPS)
        tn = total - tp - fp - fn
        train_acc = (tp + tn) / total
        train_total_acc = tp.sum() / total 

        print()
        print("Epoch: {} loss train: {:6.4f} time elapsed: {:6.1f}".format(
               e, epoch_loss_train/(i+1), time.time() - t1))

        if do_save:
            save_path = save_path + "/checkpoint-" + str(e) + '.pt'  #@Greg Add path here" '.pt'
            print('Saving model', save_path)
            torch.save(model, save_path)

        if e % 1 == 0:
            print("Training ground truth # examples")
            print(tp + fn)
            print("Training prediction # examples")
            print(tp + fp)
            print("train accuracy results: mIoU: {:6.4f} mPrecision: {:6.4f} mRecall: {:6.4f} mAcc: {:6.4f} acc: {:6.4f}".format(
                   train_iou.mean(), train_precision.mean(), train_recall.mean(), train_acc.mean(), train_total_acc))
            print("Calculating validation loss:")
            val_loss, iou, precision, recall, acc, total_acc = get_val_loss(model, loss_weights, val_loader)
            print("val loss: {:6.4f} mIoU: {:6.4f} mPrecision: {:6.4f} mRecall: {:6.4f} mAcc: {:6.4f} acc: {:6.4f}".format(
                   val_loss, iou.mean(), precision.mean(), recall.mean(), acc.mean(), total_acc))
            print("IoU for the classes are: \n", iou)
            print("Precision for the classes are: \n", precision)
            print("Recall for the classes are: \n", recall)
            print("Accuracies for the classes are: \n", acc)
            #scheduler.step(val_loss)

        print()

def main(args):
    model_id = str(time.time())
    save_path = '/home/fast_seg/FastSeg/model_checkpoints/' + model_id
    os.mkdir(save_path)
    print("Saving all output including console print outs to ", save_path)

    if args.stdout:
        sys.stdout = open(save_path + "/console.txt", "w")
    print(args)

    minibatch_size = 64
    num_classes = NUM_CLASSES

    seed = 1
    #class_weights = classes.getWeights()
    class_weights = args.w
    class_weights = torch.tensor(class_weights, dtype=torch.float, requires_grad=False, device=device)
    learning_rate = args.lr
    bn = 0
    dp = 0.1
    wd = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dset_train = coco_custom_Dataset(train_dir, length=args.n)
    dset_val = coco_custom_Dataset(val_dir, length=args.nval)
    #dset_val = coco_custom_Dataset(train_dir, length=args.nval)

    train_loader = DataLoader(dset_train, batch_size=minibatch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dset_val, batch_size=minibatch_size, shuffle=False, num_workers=0)

    

    if args.m == 0:
         model = UNet.UNet(in_channel=3, num_classes=num_classes, start_filters=args.f, num_batchnorm_layers=bn, dropout=dp)
    elif args.m == 1:
        model = U_Net_separable.U_net_separable(in_channel=3, num_classes=num_classes, start_filters=args.f, num_batchnorm_layers=bn, dropout=dp)
    elif args.m == 2:
        model = UNetSep2.UNetSep2(in_channel=3, num_classes=num_classes, start_filters=args.f, num_batchnorm_layers=bn, dropout=dp)
    elif args.m == 3:
        model = UNetSep3.UNetSep3(in_channel=3, num_classes=num_classes, start_filters=args.f, dropout=dp, nlayers=args.nl, do_sep=args.sep)
    model.to(device)

    print("Model parameters: ", count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    train_model(model, optimizer, train_loader, class_weights, val_loader, save_path, epochs=args.e, do_save=args.s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default = 1e-4,
                    help='learning rate')
    parser.add_argument('-f', type=int, default=32,
                    help='number of filters in first convolution')
    parser.add_argument('-w', type=float, nargs='+', default=[1.7,5.8,4.0],
                    help='class weights for loss function')
    parser.add_argument('-n', type=int, default=None,
                    help='# of training examples to use')
    parser.add_argument('--nval', type=int, default=None,
                    help='# of validation examples to use')
    parser.add_argument('-e', type=int, default=20,
                    help='# of epochs')
    parser.add_argument('-m', type=int, default=3,
                    help='which model to use. 0 = regular, 1 = separable, 2 = sep2, 3 = sep3')
    parser.add_argument('-s', type=int, default=1,
                    help='0 = do not save, 1 = do save model checkpoints')
    parser.add_argument('--nl', type=int, default=5,
                    help='# of layers of UNet for sep3 model')
    parser.add_argument('--sep', type=int, default=1,
                    help='boolean to use separable convolutions')
    parser.add_argument('--stdout', type=int, default=0,
                    help='boolean to rerout stdout to file')

    args = parser.parse_args()
    assert(len(args.w) == NUM_CLASSES)
    main(args)
