""" 
"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import sys, os, argparse, time, datetime
import struct
from pprint import pprint
from random import shuffle
import pdb

import numpy as np
import scipy



def _get_args():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--outlier-class', dest='outlierClass', 
		    type=int, required=True, 
		    help='Which CIFAR class (0-9) to hold out of training')

    parser.add_argument('--Z', dest='inFile', 
		    type=str, required=True,
		    help='CIFAR-10 input file name')

    parser.add_argument('--gpu', dest='gpu', 
		    type=int, default=-1, 
		    help='GPU ID to use')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='', 
		    help='(optional) overrides the snapshot directory')

    return parser.parse_args()



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


def _read_cifar10_binary(fn):
    """Loads a binary CIFAR 10 formatted file.
    
    Probably easier to use the python versions of these files, but for some
    reason having difficulty downloading these from the CIFAR website atm.

    This will produce images X with shape:  (3, 32, 32)
    If you want to view these with imshow, permute the dimensions to (32,32,3).
    """
    with open(fn, 'rb') as f:
        rawData = f.read()

    # each example is three channels of 32^2 pixels, plus one class label
    dim = 32
    dim2 = dim**2
    nChan = 3
    nBytes = dim*dim*nChan+1
    fmt = '%dB' % dim2

    yAll = []
    xAll = []
    
    for ii in xrange(0, len(rawData), nBytes):
        yAll.append(ord(rawData[ii]))
        
        red = struct.unpack(fmt, rawData[(ii+1):(ii+1+dim2)])
        green = struct.unpack(fmt, rawData[(ii+1+dim2):(ii+1+2*dim2)])
        blue = struct.unpack(fmt, rawData[(ii+1+2*dim2):(ii+1+3*dim2)])
        Xi = np.array((red,green,blue), dtype=np.uint8, order='C').reshape((1, nChan, dim, dim))
        xAll.append(Xi)

    y = np.array(yAll)
    X = np.concatenate(xAll, axis=0)
    
    return X, y



def _minibatch_generator(X, y, nBatch=100, yOmit=[], randomOrder=True):
    """Generator that returns subsets (mini-batches) of the provided data set.

    We could also add data augmentation here, but for not this isn't necessary.
    """

    # pare down to indices for labels that we do care about
    indices = [ii for ii in range(len(y)) if y[ii] not in yOmit]

    # randomize order (optional)
    if randomOrder:
        shuffle(indices)

    # Return subsets of size nBatch.
    # Note that if total number of objects is not a multiple of the mini-batch
    # size, the last mini-batch will have some data from the previous iteration
    # at the end.
    yi = np.zeros((nBatch,), dtype=y.dtype)
    Xi = np.zeros((nBatch, X.shape[1], X.shape[2], X.shape[3]), dtype=X.dtype)

    for ii in range(0, len(indices), nBatch): 
        nThisBatch = min(nBatch, len(indices)-ii)
        idx = indices[ii:ii+nThisBatch]
        
        yi[0:nThisBatch] = y[idx]
        Xi[0:nThisBatch,...] = X[idx,...]

        yield Xi, yi, nThisBatch
        


def _train_network(args):
    """ Main CNN training loop.

    Creates PyCaffe objects and calls train_one_epoch until done.
    """
    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(args.solver).read(), solverParam)

    netFn = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)

    batchDim = emlib.infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # tiles must be square
    print('[emCNN]: batch shape: %s' % str(batchDim))
   
    if args.outDir:
        outDir = args.outDir # overrides snapshot prefix
    else: 
        outDir = str(solverParam.snapshot_prefix)   # unicode -> str
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # choose a synthetic data generating function
    if args.rotateData:
        syn_func = lambda V: _xform_minibatch(V, True)
        print('[emCNN]:   WARNING: applying arbitrary rotations to data.  This may degrade performance in some cases...\n')
    else:
        syn_func = lambda V: _xform_minibatch(V, False)


    #----------------------------------------
    # Create the Caffe solver
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    solver = caffe.SGDSolver(args.solver)
    _print_net(solver.net)

    #----------------------------------------
    # Load data
    #----------------------------------------
    bs = border_size(batchDim)
    print "[emCNN]: tile radius is: %d" % bs
    
    print "[emCNN]: loading training data..."
    Xtrain, Ytrain = _load_data(args.emTrainFile,
            args.labelsTrainFile,
            tileRadius=bs,
            onlySlices=args.trainSlices,
            omitLabels=args.omitLabels)
    
    print "[emCNN]: loading validation data..."
    Xvalid, Yvalid = _load_data(args.emValidFile,
            args.labelsValidFile,
            tileRadius=bs,
            onlySlices=args.validSlices,
            omitLabels=args.omitLabels)

    #----------------------------------------
    # Do training; save results
    #----------------------------------------
    trainInfo = TrainInfo(solverParam)
    sys.stdout.flush()

    while trainInfo.iter < solverParam.max_iter: 
        print "[emCNN]: Starting epoch %d" % trainInfo.epoch
        train_one_epoch(solver, Xtrain, Ytrain, 
            trainInfo, batchDim, outDir, 
            omitLabels=args.omitLabels,
            data_augment=syn_func)

        print "[emCNN]: Making predictions on validation data..."
        Mask = np.ones(Xvalid.shape, dtype=np.bool)
        Mask[Yvalid<0] = False
        Prob = predict(solver.net, Xvalid, Mask, batchDim)

        # discard mirrored edges and form class estimates
        Yhat = np.argmax(Prob, 0) 
        Yhat[Mask==False] = -1;
        Prob = prune_border_4d(Prob, bs)
        Yhat = prune_border_3d(Yhat, bs)

        # compute some metrics
        print('[emCNN]: Validation set performance:')
        emlib.metrics(prune_border_3d(Yvalid, bs), Yhat, display=True)

        trainInfo.epoch += 1
 
    solver.net.save(str(os.path.join(outDir, 'final.caffemodel')))
    np.save(os.path.join(outDir, 'Yhat.npz'), Prob)
    scipy.io.savemat(os.path.join(outDir, 'Yhat.mat'), {'Yhat' : Prob})
    print('[emCNN]: training complete.')



#-------------------------------------------------------------------------------
# Functions for "deploying" a CNN (i.e. forward pass only)
#-------------------------------------------------------------------------------


def predict(net, X, Mask, batchDim, nMC=0):
    """Generates predictions for a data volume.

    PARAMETERS:
      X        : a data volume/tensor with dimensions (#slices, height, width)
      Mask     : a boolean tensor with the same size as X.  Only positive
                 elements will be classified.  The prediction for all 
                 negative elements will be -1.  Use this to run predictions
                 on a subset of the volume.
      batchDim : a tuple of the form (#classes, minibatchSize, height, width)

    """    
    # *** This code assumes a layer called "prob"
    if 'prob' not in net.blobs: 
        raise RuntimeError("Can't find a layer called 'prob'")

    print "[emCNN]: Evaluating %0.2f%% of cube" % (100.0*np.sum(Mask)/numel(Mask)) 

    # Pre-allocate some variables & storage.
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    nClasses = net.blobs['prob'].data.shape[1]

    # if we don't evaluate all pixels, the 
    # ones not evaluated will have label -1
    if nMC <= 0: 
        Prob = -1*np.ones((nClasses, X.shape[0], X.shape[1], X.shape[2]))
    else:
        Prob = -1*np.ones((nMC, X.shape[0], X.shape[1], X.shape[2]))
        print "[emCNN]: Generating %d MC samples for class 0" % nMC
        if nClasses > 2: 
            print "[emCNN]: !!!WARNING!!! nClasses > 2 but we are only extracting MC samples for class 0 at this time..."


    # do it
    startTime = time.time()
    cnnTime = 0
    lastChatter = -2
    it = emlib.interior_pixel_generator(X, tileRadius, batchDim[0], mask=Mask)

    for Idx, epochPct in it: 
        # Extract subtiles from validation data set 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]
            yi[jj] = 0  # this is just a dummy value

        #---------------------------------------- 
        # forward pass only (i.e. no backward pass)
        #----------------------------------------
        if nMC <= 0: 
            # this is the typical case - just one forward pass
            _tmp = time.time() 
            net.set_input_arrays(Xi, yi)
            out = net.forward() 
            cnnTime += time.time() - _tmp 
            
            # On some version of Caffe, Prob is (batchSize, nClasses, 1, 1) 
            # On newer versions, it is natively (batchSize, nClasses) 
            # The squeeze here is to accommodate older versions 
            ProbBatch = np.squeeze(out['prob']) 
            
            # store the per-class probability estimates.  
            # 
            # * On the final iteration, the size of Prob  may not match 
            #   the remaining space in Yhat (unless we get lucky and the 
            #   data cube size is a multiple of the mini-batch size).  
            #   This is why we slice yijHat before assigning to Yhat. 
            for jj in range(nClasses): 
                pj = ProbBatch[:,jj]      # get probabilities for class j 
                assert(len(pj.shape)==1)  # should be a vector (vs tensor) 
                Prob[jj, Idx[:,0], Idx[:,1], Idx[:,2]] = pj[:Idx.shape[0]]   # (*)
        else:
            # Generate MC-based uncertainty estimates
            # (instead of just a single point estimate)
            _tmp = time.time() 
            net.set_input_arrays(Xi, yi)

            # do nMC forward passes and save the probability estimate
            # for class 0.
            for ii in range(nMC): 
                out = net.forward() 
                ProbBatch = np.squeeze(out['prob']) 
                p0 = ProbBatch[:,0]      # get probabilities for class 0
                assert(len(p0.shape)==1)  # should be a vector (vs tensor) 
                Prob[ii, Idx[:,0], Idx[:,1], Idx[:,2]] = p0[:Idx.shape[0]]   # (*)
            cnnTime += time.time() - _tmp 
            

        elapsed = (time.time() - startTime) / 60.0

        if (lastChatter+2) < elapsed:  # notify progress every 2 min
            lastChatter = elapsed
            print('[emCNN]: elapsed=%0.2f min; %0.2f%% complete' % (elapsed, 100.*epochPct))
            sys.stdout.flush()

    # done
    print('[emCNN]: Total time to evaluate cube: %0.2f min (%0.2f CNN min)' % (elapsed, cnnTime/60.))
    return Prob




def _deploy_network(args):
    """ Runs Caffe in deploy mode (where there is no solver).
    """

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    netFn = str(args.network)  # unicode->str to avoid caffe API problems
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)

    batchDim = emlib.infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # tiles must be square
    print('[emCNN]: batch shape: %s' % str(batchDim))

    if args.outDir:
        outDir = args.outDir  # overrides default
    else: 
        # there is no snapshot dir in a network file, so default is
        # to just use the location of the network file.
        outDir = os.path.dirname(args.network)

    # Add a timestamped subdirectory.
    ts = datetime.datetime.now()
    subdir = "Deploy_%s_%02d:%02d" % (ts.date(), ts.hour, ts.minute)
    outDir = os.path.join(outDir, subdir)

    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    print('[emCNN]: writing results to: %s' % outDir)

    # save the parameters we're using
    with open(os.path.join(outDir, 'params.txt'), 'w') as f:
        pprint(args, stream=f)

    #----------------------------------------
    # Create the Caffe network
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    phaseTest = 1  # 1 := test mode
    net = caffe.Net(netFn, args.model, phaseTest)
    _print_net(net)

    #----------------------------------------
    # Load data
    #----------------------------------------
    bs = border_size(batchDim)
    print "[emCNN]: loading deploy data..."
    Xdeploy = _load_data(args.emDeployFile,
                         None,
                         tileRadius=bs,
                         onlySlices=args.deploySlices)
    print "[emCNN]: tile radius is: %d" % bs

    # Create a mask volume (vs list of labels to omit) due to API of emlib
    if args.evalPct < 1: 
        Mask = np.zeros(Xdeploy.shape, dtype=np.bool)
        m = Xdeploy.shape[-2]
        n = Xdeploy.shape[-1]
        nToEval = np.round(args.evalPct*m*n).astype(np.int32)
        idx = sobol(2, nToEval ,0)
        idx[0] = np.floor(m*idx[0])
        idx[1] = np.floor(n*idx[1])
        idx = idx.astype(np.int32)
        Mask[:,idx[0], idx[1]] = True
        pct = 100.*np.sum(Mask) / Mask.size
        print("[emCNN]: subsampling volume...%0.2f%% remains" % pct)
    else:
        Mask = np.ones(Xdeploy.shape, dtype=np.bool)

    #----------------------------------------
    # Do deployment & save results
    #----------------------------------------
    sys.stdout.flush()

    if args.nMC < 0: 
        Prob = predict(net, Xdeploy, Mask, batchDim)
    else:
        Prob = predict(net, Xdeploy, Mask, batchDim, nMC=args.nMC)

    # discard mirrored edges 
    Prob = prune_border_4d(Prob, bs)

    net.save(str(os.path.join(outDir, 'final.caffemodel')))
    np.save(os.path.join(outDir, 'YhatDeploy'), Prob)
    scipy.io.savemat(os.path.join(outDir, 'YhatDeploy.mat'), {'Yhat' : Prob})

    print('[emCNN]: deployment complete.')



#-------------------------------------------------------------------------------
if __name__ == "__main__":
    args = _get_args()
    print(args)

    import caffe
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    # specify CPU or GPU
    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    else:
        caffe.set_mode_cpu()

    
    # Do either training or deployment
    if args.mode == 'train':
        _train_network(args)
    else:
        _deploy_network(args)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
