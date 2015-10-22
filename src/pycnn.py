""" 
Code for training and deploying a PyCaffe network.
"""


__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import sys, re

import numpy as np



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



def metrics(Y, Yhat, display=False): 
    """
    PARAMETERS:
      Y    :  a numpy array of true class labels (vector)
      Yhat : a numpy array of estimated class labels (same size as Y)

    o Assumes any class label <0 should be ignored in the analysis.
    o Assumes all non-negative class labels are contiguous and start at 0.
      (so for binary classification, the class labels are {0,1})
    """

    # create a confusion matrix
    # yAll is all *non-negative* class labels in Y
    yAll = np.unique(Y);  yAll = yAll[yAll >= 0] 
    C = np.zeros((yAll.size, yAll.size))
    for yi in yAll:
        est = Yhat[Y==yi]
        for jj in yAll:
            C[yi,jj] = np.sum(est==jj)

    # works for arbitrary # of classes
    acc = 1.0*np.sum(Yhat[Y>=0] == Y[Y>=0]) / np.sum(Y>=0)

    # binary classification metrics (only for classes {0,1})
    nTruePos = 1.0*np.sum((Y==1) & (Yhat==1))
    precision = nTruePos / np.sum(Yhat==1)
    recall = nTruePos / np.sum(Y==1)
    f1 = 2.0*(precision*recall) / (precision+recall)

    if display: 
        for ii in range(C.shape[0]): 
            print('  class=%d    %s' % (ii, C[ii,:]))
        print('  accuracy:  %0.3f' % (acc))
        print('  precision: %0.3f' % (precision))
        print('  recall:    %0.3f' % (recall))
        print('  f1:        %0.3f' % (f1))

    return C, acc, precision, recall, f1



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

