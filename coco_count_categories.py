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

#dataDir= '/Volumes/My Passport for Mac/COCO'
dataDir= '/home/fast_seg/coco/'
#dataType='val2017'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
cats = {}
cats_rev = {}
for cat in coco.loadCats(coco.getCatIds()):
    cats[cat['id']] = cat
    cats_rev[cat['name']] = cat['id']
cats_rev['background'] = 0

imgIds = coco.getImgIds()

cat_counts = {}
for _, cat in cats.items():
    cat_counts[cat['name']] = 0
cat_counts['background'] = 0

totalArea = 0
for imgId in imgIds:

    img = coco.loadImgs([imgId])[0]

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    labeledArea = 0
    for ann in anns:
        name = cats[ann['category_id']]['name']
        cat_counts[name] += ann['area']
        labeledArea += ann['area']
    cat_counts['background'] += img['height']*img['width'] - labeledArea
    totalArea += img['height']*img['width']

cat_counts_sorted = sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)

i = 0
maxArea = cat_counts_sorted[0][1]
print("# class_id, name, coco_id, area, percent, weight") 
for x in cat_counts_sorted:
    print("{:3}, {:16}, {:3}, {:14.0f}, {:6.5f}, {:12.2f}".format(i, x[0], cats_rev[x[0]], x[1], x[1]/totalArea, maxArea/x[1]))
    i+=1
