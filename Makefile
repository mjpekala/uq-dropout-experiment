#  Use this to run UQ experiments with dropout for CIFAR-10
#  
#  1. train models, with different classes held out:
#  
#           nohup make GPU=1 HOLD_OUT=1 train &> nohup.train.1 &
#           nohup make GPU=2 HOLD_OUT=2 train &> nohup.train.2 &
#           ...
#
#  2. evaluate models on test data (after step 1 finishes):
#           make GPU=1 HOLD_OUT=1 deploy
#           ...
#
#
#  Oct 2015, mjp


PY=PYTHONPATH=~/Apps/caffe/python:./src python 

GPU=1
HOLD_OUT=0
MODEL=iter_044000.caffemodel

#-------------------------------------------------------------------------------

unittest : 
	$(PY) ./test/test_cifar_experiment.py

train : 
	$(PY) ./src/cifar_experiment.py \
		--gpu $(GPU) \
		--outlier-class $(HOLD_OUT) \
		--solver ./models/lenet_dropout_solver.prototxt \
		--out-dir ./results/HOLDOUT_$(HOLD_OUT) \
		--data ./data/data_batch_1.bin ./data/data_batch_2.bin ./data/data_batch_3.bin ./data/data_batch_4.bin ./data/data_batch_5.bin


deploy : 
	$(PY) ./src/cifar_experiment.py \
		--gpu $(GPU) \
		--network ./models/lenet_dropout_train_test.prototxt \
		--model ./results/HOLDOUT_$(HOLD_OUT)/$(MODEL) \
		--out-dir ./results/HOLDOUT_$(HOLD_OUT) \
		--data ./data/test_batch.bin
