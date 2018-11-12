from __future__ import print_function, division
import sys
sys.path.append('./cocoapi/PythonAPI')

from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
import os
import torch
#from skimage import io, transform
import imageio as io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb


class CocoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataType, transform=None, length=None):
        """
        Args:
        """
        dataDir= '/home/fast_seg/coco'
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

        self.image_path = dataDir + '/images/' + dataType + '/'

        self.transform = transform

        # initialize COCO api for instance annotations
        coco=COCO(annFile)
        self.coco = coco
        self.imgIds = coco.getImgIds()

        self.cats = {}
        for cat in coco.loadCats(coco.getCatIds()):
            self.cats[cat['id']] = cat

        if length is None:
            length = len(self.imgIds)

        self.length = length


    def __len__(self):
        # return 512
        return self.length

    def __getitem__(self, idx):

        img = self.coco.loadImgs([self.imgIds[idx]])[0]
        I = io.imread(self.image_path + img['file_name'])
        if len(I.shape) == 2:
            I = np.array([I, I, I]).transpose((1,2,0))

        annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        class_mask = np.zeros(I.shape[0:2])
        instance_mask = np.zeros(I.shape[0:2])
        for ann in anns:
            binary_mask = self.coco.annToMask(ann)
            class_mask[binary_mask==1] = ann['category_id']
            instance_mask[binary_mask==1] = ann['id']

        curr = set([self.cats[ann['category_id']]['name'] for ann in anns])
        isPerson = "person" in curr
        sample = {'image': I, "isPerson": isPerson, 'classMask': class_mask, 'instanceMask': instance_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):

    def __call__(self, sample):
        image, isPerson, classMask = sample['image'], sample['isPerson'], sample['classMask']

        tf = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = tf(image)

        return {'image': image, 'isPerson': isPerson, 'classMask': classMask}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, isPerson, classMask = sample['image'], sample['isPerson'], sample['classMask']

        #image = transform.resize(image, self.output_size, mode='constant')
        #classMask = transform.resize(classMask, self.output_size, mode='constant')
        image = scipy.misc.imresize(image, self.output_size)
        classMask = scipy.misc.imresize(classMask, self.output_size)

        return {'image': image, 'isPerson': isPerson, 'classMask': classMask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, isPerson, classMask = sample['image'], sample['isPerson'], sample['classMask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float(),
                'isPerson': torch.from_numpy(np.array(float(isPerson))).unsqueeze(0).float(),
                'classMask': torch.from_numpy(classMask)}
