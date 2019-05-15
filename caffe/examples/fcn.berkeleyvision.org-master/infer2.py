import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

caffe_root = '../../'

import sys, os 
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_logdir(sys.argv[0], os.path.abspath('./log'))

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
im = Image.open('data/pascal/VOC2007/JPEGImages/000129.jpg')
plt.imshow(im)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

### perform classification
output = net.forward()

mask = net.blobs['score'].data[0]
classed = np.argmax(mask, axis=0)

names = open(r'data/pascal/classes.txt').readlines()
all_labels = ["0", "1","2","3", "4","5", "6","7","8", "9","10", "11","12","13", "14","15", "16","17","18", "19","20"]
all_labels = [all_labels[i] + ": " + names[i] for i in range(0, len(all_labels))]

scores = np.unique(classed)
labels = [all_labels[s] for s in scores]
num_scores = len(scores)

def rescore (c):
    """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
    return np.where(scores == c)[0][0]
rescore = np.vectorize(rescore)

painted = rescore(classed)
plt.imshow(painted, cmap=plt.cm.get_cmap('jet', num_scores))

# setup legend
formatter = plt.FuncFormatter(lambda val, loc: labels[val])
plt.colorbar(ticks=range(0, num_scores), format=formatter)
plt.clim(-0.5, num_scores - 0.5)
plt.show()

