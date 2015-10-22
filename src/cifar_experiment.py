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
# some helper functions
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
    """
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
        

    
    

def _print_net(net):
    """Shows some info re. a Caffe network to stdout
    """
    for name, blobs in net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print("  %s[%d] : %s" % (name, bIdx, b.data.shape))
    for ii,layer in enumerate(net.layers):
        print("  layer %d: %s" % (ii, layer.type))
        for jj,blob in enumerate(layer.blobs):
            print("    blob %d has size %s" % (jj, str(blob.data.shape)))


            
def _load_data(xName, yName, tileRadius, onlySlices, omitLabels=None):
    """Loads data sets and does basic preprocessing.
    """
    X = emlib.load_cube(xName, np.float32)

    # usually we expect fewer slices in Z than pixels in X or Y.
    # Make sure the dimensions look ok before proceeding.
    assert(X.shape[0] < X.shape[1])
    assert(X.shape[0] < X.shape[2])

    if onlySlices: 
        X = X[onlySlices,:,:] 
    print('[emCNN]:    data shape: %s' % str(X.shape))

    X = emlib.mirror_edges(X, tileRadius)

    # Scale data to live in [0 1].
    # *** ASSUMPTION *** original data is in [0 255]
    if np.max(X) > 1:
        X = X / 255.
    print('[emCNN]:    data min/max: %0.2f / %0.2f' % (np.min(X), np.max(X)))

    # Also obtain labels file (if provided - e.g. in deploy mode
    # we may not have labels...)
    if yName: 
        Y = emlib.load_cube(yName, np.float32)

        if onlySlices: 
            Y = Y[onlySlices,:,:] 
        print('[emCNN]:    labels shape: %s' % str(Y.shape))

        # ** ASSUMPTION **: Special case code for membrane detection / ISBI volume
        yAll = np.unique(Y)
        yAll.sort()
        if (len(yAll) == 2) and (yAll[0] == 0) and (yAll[1] == 255):
            print('[emCNN]:    ISBI-style labels detected.  converting 0->1, 255->0')
            Y[Y==0] = 1;      #  membrane
            Y[Y==255] = 0;    #  non-membrane

        # Labels must be natural numbers (contiguous integers starting at 0)
        # because they are mapped to indices at the output of the network.
        # This next bit of code remaps the native y values to these indices.
        omitLabels, pctOmitted = _omit_labels(Y, omitLabels)
        Y = emlib.fix_class_labels(Y, omitLabels).astype(np.int32)

        print('[emCNN]:    yAll is %s' % str(np.unique(Y)))
        print('[emCNN]:    will use %0.2f%% of training volume' % (100.0 - pctOmitted))

        Y = emlib.mirror_edges(Y, tileRadius)

        return X, Y
    else:
        return X



#-------------------------------------------------------------------------------
# Functions for training a CNN
#-------------------------------------------------------------------------------

class TrainInfo:
    """
    Used to store/update CNN parameters over time.
    """

    def __init__(self, solverParam):
        self.param = solverParam

        self.isModeStep = (solverParam.lr_policy == u'step')

        # This code only supports some learning strategies
        if not self.isModeStep:
            raise ValueError('Sorry - I only support step policy at this time')

        if (solverParam.solver_type != solverParam.SolverType.Value('SGD')):
            raise ValueError('Sorry - I only support SGD at this time')

        # keeps track of the current mini-batch iteration and how
        # long the processing has taken so far
        self.iter = 0
        self.epoch = 0
        self.cnnTime = 0
        self.netTime = 0

        #--------------------------------------------------
        # SGD parameters.  SGD with momentum is of the form:
        #
        #    V_{t+1} = \mu V_t - \alpha \nablaL(W_t)
        #    W_{t+1} = W_t + V_{t+1}
        #
        # where W are the weights and V the previous update.
        # Ref: http://caffe.berkeleyvision.org/tutorial/solver.html
        #
        #--------------------------------------------------
        self.alpha = solverParam.base_lr  # := learning rate
        self.mu = solverParam.momentum    # := momentum
        self.gamma = solverParam.gamma    # := step factor
        self.V = {}                       # := previous values (for momentum)

        assert(self.alpha > 0)
        assert(self.gamma > 0)
        assert(self.mu >= 0)


        # XXX: weight decay
        # XXX: layer-specific weights



def _xform_minibatch(X, rotate=False, prob=0.5):
    """Synthetic data augmentation for one mini-batch.
    
    Parameters: 
       X := Mini-batch data (# slices, # channels, rows, colums) 
       
       rotate := a boolean; when true, will rotate the mini-batch X
                 by some angle in [0, 2*pi)

       prob := probability of applying any given operation

    Note: for some reason, the implementation of row and column reversals, e.g.
               X[:,:,::-1,:]
          break PyCaffe.  Numpy must be doing something under the hood 
          (e.g. changing from C order to Fortran order) to implement this 
          efficiently which is incompatible w/ PyCaffe.  
          Hence the explicit construction of X2 with order 'C' in the
          nested functions below.
    """

    def fliplr(X):
        X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
        X2[:,:,:,:] = X[:,:,::-1,:]
        return X2

    def flipud(X):
        X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
        X2[:,:,:,:] = X[:,:,:,::-1]
        return X2

    def transpose(X):
        X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
        X2[:,:,:,:] = np.transpose(X, [0, 1, 3, 2])
        return X2

    def identity(X): return X

    prob = min(1.0, prob)
    prob = max(0.0, prob)

    if rotate: 
        # rotation by an arbitrary angle 
        # Note: this is very slow!!
        # Note: this should probably be implemented at a higher level
        #       (than the individual mini-batch) so we can incorporate
        #       context rather than filling in pixels.
        angle = np.random.rand() * 360.0 
        fillColor = np.max(X) 
        X2 = scipy.ndimage.rotate(X, angle, axes=(2,3), reshape=False, cval=fillColor)
    else:
        ops = []
        ops.append( fliplr if np.random.rand() < prob else identity )
        ops.append( flipud if np.random.rand() < prob else identity )
        ops.append( transpose if np.random.rand() < prob else identity )
        shuffle(ops)
        X2 = ops[0](X)
        for op in ops[1:]:
            X2 = op(X2)

    return X2



def train_one_epoch(solver, X, Y, 
        trainInfo, 
        batchDim,
        outDir='./', 
        omitLabels=[], 
        data_augment=None):
    """ Trains a CNN for a single epoch.

    You can call multiple times - just be sure to pass in the latest
    trainInfo object each time.

    PARAMETERS:
      solver    : a PyCaffe solver object
      X         : a data volume/tensor with dimensions (#slices, height, width)
      Y         : a labels tensor with same size as X
      trainInfo : a TrainInfo object (will be modified by this function!!)
      batchDim  : the tuple (#classes, minibatchSize, height, width)
      outDir    : output directory (e.g. for model snapshots)
      omitLabels : class labels to skip during training (or [] for none)
      data_agument : synthetic data augmentation function

    """

    # Pre-allocate some variables & storage.
    #
    tileRadius = int(batchDim[2]/2)
    Xi = np.zeros(batchDim, dtype=np.float32)
    yi = np.zeros((batchDim[0],), dtype=np.float32)
    yMax = np.max(Y).astype(np.int32)
    
    tic = time.time()
    it = emlib.stratified_interior_pixel_generator(Y, tileRadius, batchDim[0], omitLabels=omitLabels) 

    for Idx, epochPct in it: 
        # Map the indices Idx -> tiles Xi and labels yi 
        # 
        # Note: if Idx.shape[0] < batchDim[0] (last iteration of an epoch) 
        # a few examples from the previous minibatch will be "recycled" here. 
        # This is intentional (to keep batch sizes consistent even if data 
        # set size is not a multiple of the minibatch size). 
        # 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]
            yi[jj] = Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ] 

        # label-preserving data transformation (synthetic data generation)
        if data_augment is not None:
            Xi = data_augment(Xi)

        # convert labels to a 4d tensor
        yiTensor = np.ascontiguousarray(yi[:, np.newaxis, np.newaxis, np.newaxis])

        assert(not np.any(np.isnan(Xi)))
        assert(not np.any(np.isnan(yi)))

        #----------------------------------------
        # one forward/backward pass and update weights
        #----------------------------------------
        _tmp = time.time()
        solver.net.set_input_arrays(Xi, yiTensor)
        out = solver.net.forward()
        assert(np.all(solver.net.blobs['data'].data == Xi))
        assert(np.all(solver.net.blobs['label'].data == yiTensor))
        solver.net.backward()

        # SGD with momentum
        for lIdx, layer in enumerate(solver.net.layers):
            for bIdx, blob in enumerate(layer.blobs): 
                if np.any(np.isnan(blob.diff)): 
                    raise RuntimeError("NaN detected in gradient of layer %d" % lIdx)
                key = (lIdx, bIdx)
                V = trainInfo.V.get(key, 0.0)
                Vnext = (trainInfo.mu * V) - (trainInfo.alpha * blob.diff)
                blob.data[...] += Vnext
                trainInfo.V[key] = Vnext

        # (try to) extract some useful info from the net
        loss = out.get('loss', None)
        acc = out.get('acc', None)

        # update run statistics
        trainInfo.cnnTime += time.time() - _tmp
        trainInfo.iter += 1
        trainInfo.netTime += (time.time() - tic)
        tic = time.time()

        #----------------------------------------
        # Some events occur on regular intervals.
        # Address these here.
        #----------------------------------------
        if (trainInfo.iter % trainInfo.param.snapshot) == 0:
            fn = os.path.join(outDir, 'iter_%06d.caffemodel' % trainInfo.iter)
            solver.net.save(str(fn))

        if trainInfo.isModeStep and ((trainInfo.iter % trainInfo.param.stepsize) ==0):
            trainInfo.alpha *= trainInfo.gamma

        if (trainInfo.iter % trainInfo.param.display) == 1: 
            print "[emCNN]: completed iteration %d of %d (epoch=%0.2f);" % (trainInfo.iter, trainInfo.param.max_iter, trainInfo.epoch+epochPct)
            print "[emCNN]:     %0.2f min elapsed (%0.2f CNN min)" % (trainInfo.netTime/60., trainInfo.cnnTime/60.)
            print "[emCNN]:     alpha=%0.4e" % (trainInfo.alpha)
            if loss: 
                print "[emCNN]:     loss=%0.3f" % loss
            if acc: 
                print "[emCNN]:     accuracy (train volume)=%0.3f" % acc
            sys.stdout.flush()
 
        if trainInfo.iter >= trainInfo.param.max_iter:
            break  # we hit max_iter on a non-epoch boundary...all done.
                
    # all finished with this epoch
    print "[emCNN]:    epoch complete."
    sys.stdout.flush()
    return loss, acc



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

    #import caffe
    #from caffe.proto import caffe_pb2
    #from google.protobuf import text_format

    ## specify CPU or GPU
    #if args.gpu >= 0:
    #    caffe.set_mode_gpu()
    #    caffe.set_device(args.gpu)
    #else:
    #    caffe.set_mode_cpu()

    
    # Do either training or deployment
    if args.mode == 'train':
        _train_network(args)
    else:
        _deploy_network(args)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
