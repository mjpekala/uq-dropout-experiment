"""Performs synthetic data augmentation on a CIFAR-10 LMDB database
   (like the one provided as an example with Caffe)

   XXX: should probably do something about correcting the mean vector...

   Example (change paths as appropriate for your caffe):

     PYTHONPATH=~/Apps/caffe/python python augment_cifar10_lmdb.py  ~/Apps/caffe/examples/cifar10/cifar10_train_lmdb  ./cifar10_train_lmdb_aug
"""

__author__ = "Mike Pekala"
__license__ = "Apache 2.0"



import sys, argparse, os.path, time
import numpy as np
import lmdb
import pdb

import caffe


NAME='aug_cifar'


def augment(X, maxJitter=2, hflip=True):
    if hflip:
        Xout = np.fliplr(X)
    else: 
        Xout = np.zeros(X.shape, dtype=X.dtype)
        Xout[...] = X[...]

    dx = int(np.round(maxJitter - np.random.rand() * 2 * maxJitter))
    dy = int(np.round(maxJitter - np.random.rand() * 2 * maxJitter))

    xv = np.arange(X.shape[1]);  xv = np.roll(xv,dx)
    yv = np.arange(X.shape[2]);  yv = np.roll(yv,dy)

    Xout[...] = Xout[:,xv,:]
    Xout[...] = Xout[:,:,yv]

    return Xout



def main(inDir, outDir, nAugment=3):

    # load the data volumes (EM image and labels, if any)
    print('[%s]: loading data from: %s' % (NAME, inDir))

    # read in the entire data set.
    X = [];  y = [];
    datum = caffe.proto.caffe_pb2.Datum()
    env = lmdb.open(inDir, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            xv = np.fromstring(datum.data, dtype=np.uint8)
            X.append(xv.reshape(datum.channels, datum.height, datum.width))
            y.append(datum.label)
    env.close()

    # create a synthetic data set
    print('[%s]: creating synthetic data...' % NAME)
    idx = 0
    datum = caffe.proto.caffe_pb2.Datum()
    env = lmdb.open(outDir, map_size=10*X[0].size*len(X)*(nAugment+1))
    with env.begin(write=True) as txn:
        for ii in range(len(y)):
            Xi = X[ii];  yi = y[ii]
            datum.channels = Xi.shape[0]
            datum.height = Xi.shape[1]
            datum.width = Xi.shape[2]
            datum.label = yi
            datum.data = Xi.tostring()
            strId = '{:08}'.format(idx)

            txn.put(strId.encode('ascii'), datum.SerializeToString())
            idx += 1

            for jj in range(nAugment):
                Xj = augment(Xi, hflip=np.mod(jj,2)==0) 
                datum.data = Xj.tostring()
                strId = '{:08}'.format(idx)
                txn.put(strId.encode('ascii'), datum.SerializeToString()) 
                idx += 1


            if np.mod(ii, 500) == 0:
                print('[%s]: Processed %d of %d images...' % (NAME, ii, len(y)))

    return 



if __name__ == "__main__":
    inDir = sys.argv[1]
    if not os.path.exists(inDir):
        raise RuntimeError('Input directory %s does not exist!' % inDir)

    outDir = sys.argv[2]
    if os.path.exists(outDir):
        raise RuntimeError('Output directory %s already exists; please move out of the way first' % outDir)

    main(inDir, outDir)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
