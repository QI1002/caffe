#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import os, sys
sys.path.insert(0, caffe_root + 'python')

import caffe

#default is cpu mode
caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

# Create target Directory if don't exist
dirname = os.path.abspath('./rcnn-log')
if not os.path.exists(dirname): os.mkdir(dirname)
caffe.set_logdir(sys.argv[0], dirname)

if (len(sys.argv) <= 1):
    df = pd.read_hdf('./rcnn_temp/det_output.h5', 'df')
else:
    if sys.argv[1].lower().endswith('csv'):
        df = pd.read_csv(sys.argv[1])
        for i in range(len(df.prediction.values)):
            df.prediction.values[i] = np.asarray([df['class'+str(x)][i] for x in range(200)])
    else:
        df = pd.read_hdf(sys.argv[1])

print(df.shape)
print(df.iloc[0])

with open('../../data/ilsvrc12/det_synset_words.txt') as f:
    labels_df = pd.DataFrame([
        {
            'synset_id': l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])
labels_df.sort_values('synset_id')
predictions_df = pd.DataFrame(np.vstack(df.prediction.values), columns=labels_df['name'])
print(predictions_df.iloc[0])

#plt.gray()
plt.matshow(predictions_df.values, cmap=plt.cm.gray)
plt.xlabel('Classes')
plt.ylabel('Windows')
plt.show()

max_s = predictions_df.max(0)
max_s.sort_values(ascending=False)
print(max_s[:10])

# Find, print, and display the top detections: person and bicycle.
i = predictions_df['person'].argmax()
j = predictions_df['bicycle'].argmax()

# Show top predictions for top detection.
f = pd.Series(df['prediction'].iloc[i], index=labels_df['name'])
print('Top detection:')
print(f.sort_values(ascending=False)[:5])
print('')

# Show top predictions for second-best detection.
f = pd.Series(df['prediction'].iloc[j], index=labels_df['name'])
print('Second-best detection:')
print(f.sort_values(ascending=False)[:5])

# Show top detection in red, second-best top detection in blue.
im = plt.imread('../images/fish-bike.jpg')
plt.imshow(im)
currentAxis = plt.gca()

det = df.iloc[i]
coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=5))

det = df.iloc[j]
coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='b', linewidth=5))

plt.show()

def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

scores = predictions_df['bicycle']
windows = df[['xmin', 'ymin', 'xmax', 'ymax']].values
dets = np.hstack((windows, scores[:, np.newaxis]))

plt.imshow(im)
currentAxis = plt.gca()
colors = ['r', 'b', 'y']
for c, det in zip(colors, dets[:3]):
    currentAxis.add_patch(
        plt.Rectangle((det[0], det[1]), det[2]-det[0], det[3]-det[1],
        fill=False, edgecolor=c, linewidth=5)
    )
print 'scores(before nms):', dets[:3, 4]

plt.show()

nms_dets = nms_detections(dets)

plt.imshow(im)
currentAxis = plt.gca()
colors = ['r', 'b', 'y']
for c, det in zip(colors, nms_dets[:3]):
    currentAxis.add_patch(
        plt.Rectangle((det[0], det[1]), det[2]-det[0], det[3]-det[1],
        fill=False, edgecolor=c, linewidth=5)
    )
print 'scores(after nms):', nms_dets[:3, 4]

plt.show()
