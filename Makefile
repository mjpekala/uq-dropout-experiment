#  Use this to run UQ experiments with dropout for CIFAR-10
#  
#  EXAMPLE: train a few models, holding out a different class
#           each time:
#
#           nohup make GPU=1 HOLD_OUT=1 train &> nohup.train.1 &
#           nohup make GPU=2 HOLD_OUT=2 train &> nohup.train.2 &
#           nohup make GPU=3 HOLD_OUT=3 train &> nohup.train.3 &
#           nohup make GPU=4 HOLD_OUT=4 train &> nohup.train.4 &
#           nohup make GPU=5 HOLD_OUT=5 train &> nohup.train.5 &
#           nohup make GPU=6 HOLD_OUT=6 train &> nohup.train.6 &
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
