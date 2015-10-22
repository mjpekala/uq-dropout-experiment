#  Use this to run UQ experiments with dropout for CIFAR-10
#
#  Oct 2015, mjp


PY=PYTHONPATH=~/Apps/caffe/python:./src python 

GPU=1
HOLD_OUT=0

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
