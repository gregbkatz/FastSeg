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

cats = coco.loadCats(coco.getCatIds())
pdb.set_trace()
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# display COCO categories and supercategories
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# Redefine cats as a dict
cats = {}
for cat in coco.loadCats(coco.getCatIds()):
    cats[cat['id']] = cat

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

while True:
    imgId = np.random.choice(imgIds)

    img = coco.loadImgs([imgId])[0]

    # load and display image
    I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))

    plt.axis('off')
    plt.imshow(I)

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    curr = set([cats[ann['category_id']]['name'] for ann in anns])
    plt.title(curr)

    coco.showAnns(anns)

    plt.show()
