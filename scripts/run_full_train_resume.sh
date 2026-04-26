#!/bin/bash
set -e
WS=$HOME/workspace
LOGDIR=$WS/logs
mkdir -p $LOGDIR
cd $WS/code/OpenOneRec/pretrain
RESUME_STEP=${1:-21500}
sudo docker run --rm --gpus all --ipc=host --name onerec_train \
    -v $WS:$WS -w $WS/code/OpenOneRec/pretrain \
    -e MAX_LENGTH=32768 -e MINIBATCH_SIZE=16384 \
    -e NUM_STEPS=40000 -e NUM_WARMUP=3000 -e SAVE_PER_STEP=500 \
    -e RESUME_FROM=$WS/outputs/sft/step$RESUME_STEP -e RESUME_TAG=global_step$RESUME_STEP \
    onerec-train:latest \
    bash examples/posttrain_sft_1gpu.sh 2>&1 | tee -a $LOGDIR/train.log
