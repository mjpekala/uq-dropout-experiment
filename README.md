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

