"""Unit tests.

To run (from pwd):
    PYTHONPATH=../src python test_cifar10_experiment.py
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import unittest
import os
import numpy as np

import cifar_experiment as ce



class TestCifar10Experiment(unittest.TestCase):
    def test_minibatch_generator(self):
        fn = os.path.join('..', 'data', 'data_batch_1.bin')

        X, y = ce._read_cifar10_binary(fn)
        yAll = np.zeros((10,))
        
        for X, y, n in ce._minibatch_generator(X, y, nBatch=100, yOmit=[0,]):
            for label in np.unique(y):
                yAll[label] += np.sum(y == label)

        self.assertTrue(yAll[0] == 0)
        for ii in range(1,10):
            self.assertTrue(yAll[ii] > 100)
            
            
        

        
if __name__ == "__main__":
    unittest.main()


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
