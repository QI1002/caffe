#!/usr/bin/env python
"""
read *.npy which is the output of classify.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
import sys
import caffe

def vis_detections(image_path, class_name, bbox, score):

    im = caffe.io.load_image(image_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    ax.add_patch(
        plt.Rectangle((bbox['xmin'], bbox['ymin']),
                      bbox['xmax'] - bbox['xmin'],
                      bbox['ymax'] - bbox['ymin'], fill=False, 
                      edgecolor='red', linewidth=3.5)
        )

    ax.set_title(('{} is detected (score={:.3f})').format(class_name, score),
                  fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if len(sys.argv) < 2:
    print 'no argument'
    sys.exit()

if not os.path.isfile(sys.argv[1]) or not os.access(sys.argv[1], os.R_OK):    
    print 'file', sys.argv[1], 'is missing or is not readable'
    sys.exit()

df = pd.read_csv(sys.argv[1])

filename = 'data/ilsvrc12/det_synset_words.txt'
if not os.path.isfile(filename) or not os.access(filename, os.R_OK):
    print 'file', filename, 'is missing or is not readable'
    sys.exit()

max_value_array = list()
max_index_array = list()

for j in range(0, len(df['filename'])):
    max_index = 0
    max_value = -1000.0
    for i in range(0, 199):
        if df['class'+str(i)][j] > max_value:
            max_value = df['class'+str(i)][j]
            max_index = i

        max_value_array.append(max_value)
        max_index_array.append(max_index)

score_idx = np.argsort(max_value_array)
score_idx = score_idx[::-1]

for j in score_idx:

    max_value = max_value_array[j]
    max_index = max_index_array[j]

    i = 0
    f = file(filename)
    while True:
        line = f.readline()
        if len(line) == 0 or i == max_index:
            break
        i = i + 1

    f.close()

    bbox = dict()
    bbox['xmin'] = df['xmin'][j]
    bbox['ymin'] = df['ymin'][j]
    bbox['xmax'] = df['xmax'][j]
    bbox['ymax'] = df['ymax'][j]

    vis_detections(df['filename'][j], line.rstrip(), bbox, df['class'+str(i)][j])
    print '{0:.4f}'.format(max_value), '-', line.rstrip(), 'range[{}]={},{},{},{}'.format(
          j, df['xmin'][j], df['ymin'][j], df['xmax'][j], df['ymax'][j]) 
    plt.show()
