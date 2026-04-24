#!/bin/bash
# Single-GPU adaptation of posttrain_sft.sh (strips mpirun, MPI, Kuaishou-internal env)
set -e
PRETRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PRETRAIN_DIR"
export PYTHONPATH="$PRETRAIN_DIR:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:-/home/v-daqianding/workspace/data/hf_models/OneRec-1.7B-pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/v-daqianding/workspace/outputs/sft}"
mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8499
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_LENGTH=${MAX_LENGTH:-8192}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-4096}
NUM_STEPS=${NUM_STEPS:-5000}
NUM_WARMUP=${NUM_WARMUP:-500}
SAVE_PER_STEP=${SAVE_PER_STEP:-500}
LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-1e-4}

echo "=== launching single-GPU SFT ==="
echo "MODEL_DIR=$MODEL_DIR OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_LENGTH=$MAX_LENGTH MINIBATCH=$MINIBATCH_SIZE NUM_STEPS=$NUM_STEPS WARMUP=$NUM_WARMUP"
nvidia-smi -L || true

python3 recipes/train_qwen3.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_config examples/dataset_config/sft.json \
    --use_tie_weights \
    --model_class Qwen3ForCausalLM \
    --max_length $MAX_LENGTH \
    --learning_rate $LR \
    --min_lr $MIN_LR \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --num_warmup_steps $NUM_WARMUP \
    --num_training_steps $NUM_STEPS \
    --save_checkpoint_per_step $SAVE_PER_STEP \
    --minibatch_size $MINIBATCH_SIZE \
    --logging_per_step 5 \
    --use_fp32_weight \
    --seed 19260817 \
    --enable_gradient_checkpointing \
    --use_chunked_loss_computer \
    ${RESUME_FROM:+--resume_from $RESUME_FROM --resume_from_tag $RESUME_TAG --resume_training_state}
