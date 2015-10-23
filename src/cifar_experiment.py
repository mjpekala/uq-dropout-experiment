""" 
  This code supports experiments using the approach of [G&G] 
  to estimate uncertainty for held-out examples for the CIFAR-10
  data set.

  See the Makefile for examples of how to use this script.

  Note: since we are not currently doing any special synthetic
  data augmentation on-the-fly, could probably use the SGD 
  API that is available in PyCaffe (vs implementing a 
  limited subset of SGD here).

  REFERENCES:
    o  Gal & Ghahramani "Dropout as a Bayesian Approximation: 
       Representing Model Uncertainty in Deep Learning"
"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"



import sys, os, argparse, time, datetime, re
import struct
from pprint import pprint
from random import shuffle
#import pdb

import numpy as np
from scipy.io import savemat




def _get_args():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()

    #--------------------------------------------------
    # these args are for TRAIN mode
    #--------------------------------------------------
    parser.add_argument('--solver', dest='solver', 
		    type=str, default=None, 
		    help='The caffe solver file to use (TRAINING mode)')

    parser.add_argument('--outlier-class', dest='outlierClass', 
		    type=int, default=-1,
		    help='Which CIFAR class (0-9) to hold out (TRAINING mode)')

    #--------------------------------------------------
    # these args are for TEST mode
    #--------------------------------------------------
    parser.add_argument('--network', dest='network', 
		    type=str, default=None, 
		    help='The caffe network file to use (DEPLOY mode)')

    parser.add_argument('--model', dest='model', 
		    type=str, default=None, 
		    help='The trained caffe model to use (DEPLOY mode)')

    #--------------------------------------------------
    # these args are mode-independent
    #--------------------------------------------------
    parser.add_argument('--data', dest='inFiles', 
		    type=argparse.FileType('r'), required=True,
                    nargs='+',
		    help='CIFAR-10 input file name(s)')

    parser.add_argument('--gpu', dest='gpu', 
		    type=int, default=-1, 
		    help='GPU ID to use')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='', 
		    help='(optional) overrides the snapshot directory')

    parser.add_argument('--num-samp', dest='nSamp', 
		    type=int, default=30, 
		    help='Number of stochastic forward passes to use')


    args = parser.parse_args()


    if args.solver is not None:
        args.mode = 'train'
    elif args.network is not None:
        args.mode = 'deploy'
    else:
        raise RuntimeError('you must specify either a solver or a network')


    return args


#-------------------------------------------------------------------------------

def _read_cifar10_binary(fileObj):
    """Loads a binary CIFAR 10 formatted file.
    
    Probably easier to use the python versions of these files, but for some
    reason having difficulty downloading these from the CIFAR website atm.

    This will produce images X with shape:  (3, 32, 32)
    If you want to view these with imshow, permute the dimensions to (32,32,3).
    """
    rawData = fileObj.read()

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




def print_net(net):
    """Shows some info re. a Caffe network to stdout

    net : a PyCaffe Net object.
    """
    for name, blobs in net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print("  %s[%d] : %s" % (name, bIdx, b.data.shape))
    for ii,layer in enumerate(net.layers):
        print("  layer %d: %s" % (ii, layer.type))
        for jj,blob in enumerate(layer.blobs):
            print("    blob %d has size %s" % (jj, str(blob.data.shape)))



def infer_data_dimensions(netFn):
    """Determine the size of the Caffe input data tensor.

    There may be a cleaner way to do this through the pycaffe API (e.g. via the
    network parameters protobuf object).
    """
    with open(netFn, 'r') as f:
        contents = "".join(f.readlines())

    dimNames = ['batch_size', 'channels', 'height', 'width']
    dimensions = np.zeros((4,), dtype=np.int32)

    for ii, dn in enumerate(dimNames):
        pat = r'%s:\s*(\d+)' % dn
        mo = re.search(pat, contents)
        if mo is None:
            raise RuntimeError('Unable to extract "%s" from network file "%s"' % (dn, netFn))
        dimensions[ii] = int(mo.groups()[0])
        
    return dimensions



def _minibatch_generator(X, y, nBatch, yOmit=[], randomOrder=True):
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



def _eval_performance(Prob, yTrue): 
    """
      Prob  : a tensor with dimensions (#examples, #classes, #samples)
      yTrue : a vector with dimensions (#examples,)
    """
    assert(len(Prob.shape) == 3)

    Mu = np.mean(Prob, axis=2)
    Yhat = np.argmax(Mu, axis=1)
    acc = 100.0 * np.sum(Yhat == yvalid)  / yvalid.shape 

    # TODO: per-class accuracy
    return acc


#-------------------------------------------------------------------------------
# Training mode
#-------------------------------------------------------------------------------

class TrainInfo:
    """
    Parameters used during CNN training
    (some of these change as training progresses...)
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



def _train_network(XtrainAll, ytrainAll, Xvalid, yvalid, args):
    """ Main CNN training loop.
    """

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(args.solver).read(), solverParam)

    netFn = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)

    batchDim = infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # images must be square
    print('[train]: batch shape: %s' % str(batchDim))

    # create output directory if it does not already exist
    if args.outDir:
        outDir = args.outDir # overrides snapshot prefix
    else: 
        outDir = str(solverParam.snapshot_prefix)   # unicode -> str
    if not os.path.isdir(outDir):
        os.makedirs(outDir)


    #----------------------------------------
    # Create the Caffe solver
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    solver = caffe.SGDSolver(args.solver)
    print_net(solver.net)
    sys.stdout.flush()

    trainInfo = TrainInfo(solverParam)
    all_done = lambda x: x >= solverParam.max_iter


    #----------------------------------------
    # Do training
    #----------------------------------------
    tic = time.time()

    while not all_done(trainInfo.iter):

        #----------------------------------------
        # This loop is over training files
        #----------------------------------------
        for ii in range(len(ytrainAll)):
            if all_done(trainInfo.iter): break

            Xi = XtrainAll[ii];  yi = ytrainAll[ii]

            #----------------------------------------
            # this loop is over examples in the current file
            #----------------------------------------
            for Xb, yb, nb in _minibatch_generator(Xi, yi, batchDim[0], yOmit=[args.outlierClass,]):
                if all_done(trainInfo.iter): break 
                
                # convert labels to a 4d tensor 
                ybTensor = np.ascontiguousarray(yb[:, np.newaxis, np.newaxis, np.newaxis]) 
                assert(not np.any(np.isnan(Xb))) 
                assert(not np.any(np.isnan(yb))) 
                
                #---------------------------------------- 
                # one forward/backward pass and update weights 
                #---------------------------------------- 
                _tmp = time.time() 
                solver.net.set_input_arrays(Xb, ybTensor) 
                out = solver.net.forward() 
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
                # Handle those here...
                #---------------------------------------- 

                # save model snapshot
                if (trainInfo.iter % trainInfo.param.snapshot) == 0: 
                    fn = os.path.join(outDir, 'iter_%06d.caffemodel' % trainInfo.iter) 
                    solver.net.save(str(fn)) 
                   
                # update learning rate
                if trainInfo.isModeStep and ((trainInfo.iter % trainInfo.param.stepsize) == 0): 
                    trainInfo.alpha *= trainInfo.gamma 
               
                # display progress to stdout
                if (trainInfo.iter % trainInfo.param.display) == 1: 
                    print "[train]: completed iteration %d of %d" % (trainInfo.iter, trainInfo.param.max_iter) 
                    print "[train]:     %0.2f min elapsed (%0.2f CNN min)" % (trainInfo.netTime/60., trainInfo.cnnTime/60.) 
                    print "[train]:     alpha=%0.4e" % (trainInfo.alpha) 
                    if loss: 
                        print "[train]:     loss=%0.3f" % loss 
                    if acc: 
                        print "[train]:     Accuracy (train volume)=%0.3f" % acc
                    sys.stdout.flush()


        #----------------------------------------
        # Evaluate on the held-out data set
        #----------------------------------------
        if all_done(trainInfo.iter): 
            print "[train]: Max number of iterations reached (%d)" % trainInfo.iter
        else:
            print "[train]: Completed epoch (iter=%d);" % trainInfo.iter
        print "[train]: evaluating validation data..."

        Prob = predict(solver.net, Xvalid, batchDim, nSamp=args.nSamp)
        acc = _eval_performance(Prob, yvalid)
        print "[train]: accuracy on validation data set: %0.3f" % acc


    print('[train]: training complete.')
    print "[train]:     %0.2f min elapsed (%0.2f CNN min)" % (trainInfo.netTime/60., trainInfo.cnnTime/60.) 
    return Prob


#-------------------------------------------------------------------------------
# Deploy mode
#-------------------------------------------------------------------------------

def _deploy_network(X, y, args):
    """ Runs Caffe in deploy mode (where there is no solver).
    """

    #----------------------------------------
    # parse information from the prototxt files
    #----------------------------------------
    netFn = str(args.network)  # unicode->str to avoid caffe API problems
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(), netParam)

    batchDim = infer_data_dimensions(netFn)
    assert(batchDim[2] == batchDim[3])  # tiles must be square
    print('[deploy]: batch shape: %s' % str(batchDim))

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
    print('[deploy]: writing results to: %s' % outDir)

    # save the parameters we're using
    with open(os.path.join(outDir, 'params.txt'), 'w') as f:
        pprint(args, stream=f)

    #----------------------------------------
    # Create the Caffe network
    # Note this assumes a relatively recent PyCaffe
    #----------------------------------------
    phaseTest = 1  # 1 := test mode
    net = caffe.Net(netFn, args.model, phaseTest)
    print_net(net)

    #----------------------------------------
    # Do deployment & save results
    #----------------------------------------
    sys.stdout.flush() 
    
    Prob = predict(net, X, batchDim, nSamp=args.nSamp) 
    acc = _eval_performance(Prob, y) 
    print "[train]: accuracy on test data set: %0.3f" % acc

    np.save(os.path.join(outDir, 'ProbDeploy'), Prob)
    savemat(os.path.join(outDir, 'ProbDeploy.mat'), {'Prob' : Prob})

    print('[deploy]: deployment complete.')


#-------------------------------------------------------------------------------
# Both train and deploy make use of Caffe forward passes
#-------------------------------------------------------------------------------

def predict(net, X, batchDim, nSamp=30):
    """Generates predictions for a data volume.

    PARAMETERS:
      X        : a data volume/tensor with dimensions (#slices, height, width)
      batchDim : a tuple of the form (#classes, minibatchSize, height, width)

    """    
    # *** This code assumes a softmax layer called "prob" with 
    # a single output of the same name
    if 'prob' not in net.blobs: 
        raise RuntimeError("Can't find a layer called 'prob'")


    # Pre-allocate some variables & storage.
    nClasses = net.blobs['prob'].data.shape[1]
    Prob = -1*np.ones( (X.shape[0], nClasses, nSamp) )
    Prob_mb = np.zeros( (batchDim[0], nClasses, nSamp) )
    yDummy = np.zeros((X.shape[0],), dtype=np.float32)

    # do it
    startTime = time.time()
    cnnTime = 0
    numEvaluated = 0
    lastChatter = -2

    for Xb, yb, nb in _minibatch_generator(X, yDummy, batchDim[0], yOmit=[], randomOrder=False):
        # convert labels to a 4d tensor 
        ybTensor = np.ascontiguousarray(yb[:, np.newaxis, np.newaxis, np.newaxis]) 
        assert(not np.any(np.isnan(Xb))) 
        assert(not np.any(np.isnan(yb))) 
                
        # Generate MC-based uncertainty estimates
        # (instead of just a single point estimate)
        _tmp = time.time() 
        net.set_input_arrays(Xb, yb)

        # do nMC forward passes and save the probability estimate
        # for class 0.
        for ii in range(nSamp): 
            out = net.forward() 
            Prob_mb[:,:,ii] = np.squeeze(out['prob'])
        cnnTime += time.time() - _tmp 
      
        # put the estimates for this mini-batch into the larger overall 
        # return value.
        Prob[numEvaluated:(numEvaluated+nb),:,:] = Prob_mb[0:nb,:,:]


        elapsed = (time.time() - startTime) / 60.0
        numEvaluated += nb

        if (lastChatter+2) < elapsed:  # notify progress every 2 min
            lastChatter = elapsed
            print('[predict]: elapsed=%0.2f min; %0.2f%% complete' % (elapsed, 100.*numEvaluated/X.shape[0]))
            sys.stdout.flush()

    # done
    print('[predict]: Total time to deploy: %0.2f min (%0.2f CNN min)' % (elapsed, cnnTime/60.))
    
    return Prob



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

    # load the data set
    yall = []
    Xall = []
    for ii in range(len(args.inFiles)):
        X,y = _read_cifar10_binary(args.inFiles[ii])
        yall.append(y.astype(np.float32))
        Xall.append(X.astype(np.float32)/255.)

    print('[info]: read %d CIFAR-10 files' % len(yall))


    # TRAIN mode
    if args.mode == 'train':
        if len(yall) <= 1:
            raise RuntimeError('for training, expect at least 2 files!')
        Prob = _train_network(Xall[0:-1], yall[0:-1], Xall[-1], yall[-1], args)

        np.save(os.path.join(args.outDir, 'P_valid'), Prob)
        savemat(os.path.join(args.outDir, 'P_valid.mat'), {'Prob' : Prob})

    # DEPLOY mode
    else:
        if len(yall) > 1:
            raise RuntimeError('for deployment, expect only 1 file!')
        Prob = _deploy_network(Xall[0], yall[0], args)

    print('[info]: all finished!')


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
