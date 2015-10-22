""" 
Code for training and deploying a PyCaffe network.
"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"




def _print_net(net):
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

            
#-------------------------------------------------------------------------------
# Functions for training a CNN
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

