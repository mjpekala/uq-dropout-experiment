
PY=PYTHONPATH=~/Apps/caffe/python:./src python 
GPU=1

HOLD_OUT=0


unittest : 
	$(PY) ./test/test_cifar_experiment.py

train : 
	$(PY) ./src/cifar_experiment.py \
		--data ./data/data_batch_1.bin ./data/data_batch_2.bin \
		--solver ./models/lenet_dropout_solver.prototxt \
		--outlier-class $(HOLD_OUT) \
		--out-dir ./results/ho$(HOLD_OUT) \
		--gpu $(GPU)
