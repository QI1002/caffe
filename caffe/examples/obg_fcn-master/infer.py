import numpy as np
from PIL import Image

caffe_root = '../../'

import sys, os 
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_logdir(sys.argv[0], os.path.abspath('./log'))

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
im = Image.open('data/pascal/VOC2007/JPEGImages/000129.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('fcn8s/deploy.prototxt', 'fcn8s/fcn8s-heavy-40k.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

#print str(out)
class_num = {}
for i in range(0, out.shape[0]):
    for j in range(0, out.shape[1]):
        if out[i][j] in class_num.keys(): 
            class_num[out[i][j]] += 1
        else:
            class_num[out[i][j]] = 0

print str(class_num)
