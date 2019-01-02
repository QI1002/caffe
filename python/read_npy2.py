#!/usr/bin/env python
"""
read *.npy which is the output of classify.py
"""
import numpy as np
import os 
import os.path
import sys

if len(sys.argv) < 2:
    print 'no argument'
    sys.exit()

if not os.path.isfile(sys.argv[1]) or not os.access(sys.argv[1], os.R_OK):
    print 'file ', sys.argv[1], ' is missing or is not readable'
    sys.exit()

array_loaded = np.load(sys.argv[1])
#print 'Loaded: ', array_loaded

filename = '../data/ilsvrc12/det_synset_words.txt'
if not os.path.isfile(filename) or not os.access(filename, os.R_OK):
    print 'file ', filename, ' is missing or is not readable'
    sys.exit()

for j in range(0, 5):
    max_index = 0
    max_value = -1000.0
    for i in range(0, 199):
        if array_loaded[0][i] > max_value:
            max_value = array_loaded[0][i]
            max_index = i

    i = 0

    f = file(filename)
    while True:
        line = f.readline()
        if len(line) == 0 or i == max_index:
            break
        i = i + 1

    f.close()

    print '{0:.4f}'.format(max_value), '-', line.rstrip()
    array_loaded[0][max_index] = -1000.0
