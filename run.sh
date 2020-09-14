#!/bin/bash
PYTHON='python3'

MODEL='WNET' #{'WNET', 'RL'}
## Train
if [[ "$1" == "--train" ]];then
	echo "========== TRAIN =========="
	$PYTHON main_train.py --model=$MODEL
fi