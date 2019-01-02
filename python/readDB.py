#!/usr/bin/env python

import leveldb
import binascii
import numpy as np
import caffe
import sys
from caffe.proto import caffe_pb2

# parse input argument
dbName = sys.argv[1]

# open leveldb files 

db = leveldb.LevelDB(dbName)

# get db iterator

it = db.RangeIter()

for key, value in it:
    # convert string to datum
    
    datum = caffe_pb2.Datum.FromString(db.Get(key))

    # convert datum to numpy string 
    arr = caffe.io.datum_to_array(datum)[0]

    i = 0
    tmpS = ''

    # convert to svm format 
    for i in range(0, len(arr)):
        tmpS += str(i+1) + ':' + str(arr[i].tolist()[0]) + ' '
    print tmpS

