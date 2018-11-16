import sys
sys.path
sys.path.append('./cocoapi/PythonAPI')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pdb

dataDir= '/Volumes/My Passport for Mac 1/COCO'
dataType='val2017'
# dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
cats = {}
for cat in coco.loadCats(coco.getCatIds()):
    cats[cat['id']] = cat

imgIds = coco.getImgIds()

cat_counts = {}
for _, cat in cats.items():
    cat_counts[cat['name']] = 0

for imgId in imgIds:

    img = coco.loadImgs([imgId])[0]

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    curr = set([cats[ann['category_id']]['name'] for ann in anns])
    for x in curr:
        cat_counts[x] += 1


cat_counts_sorted = sorted(cat_counts.items(), key=lambda kv: kv[1])

L = len(imgIds)
for x in cat_counts_sorted:
    print(x[0], x[1]/L)