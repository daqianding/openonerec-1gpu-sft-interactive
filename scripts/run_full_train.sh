#!/bin/bash
set -e
WS=$HOME/workspace
LOGDIR=$WS/logs
mkdir -p $LOGDIR
cd $WS/code/OpenOneRec/pretrain

# full training: 30000 steps @ 4.5s = 37.5h, plenty of margin within 50h budget
sudo docker run --rm --gpus all --ipc=host --name onerec_train \
    -v $WS:$WS -w $WS/code/OpenOneRec/pretrain \
    -e MAX_LENGTH=32768 -e MINIBATCH_SIZE=16384 \
    -e NUM_STEPS=30000 -e NUM_WARMUP=3000 -e SAVE_PER_STEP=500 \
    onerec-train:latest \
    bash examples/posttrain_sft_1gpu.sh 2>&1 | tee $LOGDIR/train.log
