# uq-dropout-experiment

This repository hosts an experiment for working with the uncertainty quantification approach of Gal & Ghahramani in their paper ["Dropout as Bayesian Approximation: Representing Model Uncertainty in Deep Learning"](http://arxiv.org/abs/1506.02142).

*This code is in an experimental state and subject to change.*


## Quick start

### Configuring Caffe
This software requires Caffe (tested with version 1.0, release candidate 2).

- https://github.com/BVLC/caffe/releases

Note that the following modifications to Caffe are required:

- Fix the memory leak in the memory data layer (see https://github.com/BVLC/caffe/issues/2334).  Also initialize labels_ and data_ in the MemoryDataLayer (include/caffe/data_layers.hpp) if you want the Caffe unit tests to pass.
```
   Dtype* data_ = NULL;
   Dtype* labels_ = NULL;
```
- Apply the patch described here: https://github.com/yaringal/DropoutUncertaintyCaffeModels (permits stochastic forward passes at "deploy" time).  Note that I called the new parameter "do_mc" instead of "sample_weights_test".


### Training and Generating Estimates
You'll first need to get the CIFAR-10 data set, e.g.:
```
    cd ./data
    ./getcifar.sh
    gunzip cifar-10-binary.tar.gz
```

Once you have the data, you can use the provided [Makefile](./Makefile) to train models.  For example, to train models with different classes held out you can do
```
    nohup make GPU=1 HOLD_OUT=1 train &> nohup.train.1 &
    nohup make GPU=2 HOLD_OUT=2 train &> nohup.train.2 &
    ...
	```

Once these models have trained, you can evaluate (i.e. carry out stochastic forward passes) via
```
    make GPU=1 HOLD_OUT=1 deploy
    ...
```

Deploying is usually much faster than training, hence the lack of nohup here.
You'll can modify these examples to use different Caffe configuration files or enable/disable other options.

Note also this code currently only supports a small set of Caffe solvers/parameters.  The code could (should?) be changed to support the PyCaffe solver API (assuming it is compatible with the memory data layer); alternatively, there's a slightly improved implementation of the Python-based solver availble [here](https://github.com/mjpekala/faster-membranes).
