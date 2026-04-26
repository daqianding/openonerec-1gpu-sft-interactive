#!/bin/bash
export HF_TOKEN=$(cat ~/.hf_token)
export PATH=$HOME/.local/bin:$PATH
WS=$HOME/workspace
FEISHU=https://open.feishu.cn/open-apis/bot/v2/hook/cec8c727-b5a8-46e4-995a-051bfa2ea9a2
HF_REPO=dqding/OneRec-1.7B-sft-interactive-1gpu
SRC_HF=$WS/data/hf_models/OneRec-1.7B-pretrain
PUSHED=$WS/backup_pushed.txt
PUSHED_FULL=$WS/backup_pushed_full.txt
touch $PUSHED $PUSHED_FULL
LAST_LOG_STEP=0
WEIGHTS_INTERVAL=2000
WEIGHTS_FROM=5000
FULL_MILESTONES=(15000 30000)
KEEP_RECENT=2
USR=$(id -u):$(id -g)

notify(){ curl -s -X POST -H "Content-Type: application/json" $FEISHU -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$1\"}}" >/dev/null||true; }

retry_upload() {
    local SRC=$1 DEST=$2 MSG=$3
    for try in 1 2 3 4 5; do
        if hf upload $HF_REPO $SRC $DEST --repo-type model --commit-message "$MSG" 2>&1 | tail -3; then
            return 0
        fi
        notify "[OneRec][WARN] upload retry $try for $DEST"
        sleep $((try * 30))
    done
    notify "[OneRec][ERR] upload FAILED after 5 retries: $DEST"
    return 1
}

convert_and_push_weights() {
    local STEP=$1
    local CKPT_DIR=$WS/outputs/sft/step$STEP
    local TMP_OUT=$WS/tmp_hf_step$STEP
    sudo rm -rf $TMP_OUT 2>/dev/null
    notify "[OneRec] converting step$STEP -> HF safetensors"
    sudo docker run --rm -u $USR --ipc=host \
        -v $WS:$WS -w $WS/code/OpenOneRec/pretrain \
        onerec-train:latest \
        python3 tools/model_converter/convert_checkpoint_to_hf.py \
            --checkpoint_dir $CKPT_DIR/global_step$STEP \
            --output_dir $TMP_OUT \
            --source_hf_model_path $SRC_HF \
            --use_safetensor --dtype bf16 --max_gb_per_shard 5 \
        2>&1 | tail -10
    if [ ! -f $TMP_OUT/config.json ]; then
        notify "[OneRec][ERR] convert step$STEP failed"
        sudo rm -rf $TMP_OUT 2>/dev/null
        return 1
    fi
    SIZE=$(du -sh $TMP_OUT | awk "{print \$1}")
    notify "[OneRec] converted step$STEP ($SIZE), uploading"
    if retry_upload $TMP_OUT weights/step$STEP "weights step$STEP (bf16 safetensors)"; then
        echo $STEP >> $PUSHED
        notify "[OneRec] ✅ pushed weights/step$STEP"
    fi
    sudo rm -rf $TMP_OUT 2>/dev/null || rm -rf $TMP_OUT 2>/dev/null
}

push_full_ckpt() {
    local STEP=$1
    notify "[OneRec] pushing FULL distcp ckpt step$STEP (22GB)"
    if retry_upload $WS/outputs/sft/step$STEP ckpt/step$STEP "full distcp ckpt step$STEP"; then
        echo $STEP >> $PUSHED_FULL
        notify "[OneRec] ✅ pushed full ckpt/step$STEP"
    fi
}

cleanup_local_ckpts() {
    # Keep: latest KEEP_RECENT ckpts + any FULL_MILESTONES
    local ALL=($(ls -1 $WS/outputs/sft/ | grep -E "^step[0-9]+$" | sed "s/step//" | sort -n))
    local N=${#ALL[@]}
    if [ $N -le $KEEP_RECENT ]; then return; fi
    local KEEP_FROM=$((N - KEEP_RECENT))
    for i in $(seq 0 $((N-1))); do
        local S=${ALL[$i]}
        # Always keep recent
        if [ $i -ge $KEEP_FROM ]; then continue; fi
        # Always keep milestones
        local IS_MS=0
        for M in "${FULL_MILESTONES[@]}"; do
            if [ "$S" = "$M" ]; then IS_MS=1; break; fi
        done
        if [ $IS_MS -eq 1 ]; then continue; fi
        # Only delete if weights already pushed (safety)
        if grep -q "^$S$" $PUSHED 2>/dev/null || [ $((S % WEIGHTS_INTERVAL)) -ne 0 ]; then
            echo "[cleanup] removing local step$S"
            sudo rm -rf $WS/outputs/sft/step$S
        fi
    done
}

while true; do
    sleep 300
    LATEST_DIR=$(ls -1 $WS/outputs/sft/ 2>/dev/null | grep -E "^step[0-9]+$" | sed "s/step//" | sort -n | tail -1)
    STEP=${LATEST_DIR:-0}

    if [ "$STEP" -ge $WEIGHTS_FROM ]; then
        for CAND in $(ls -1 $WS/outputs/sft/ | grep -E "^step[0-9]+$" | sed "s/step//" | sort -n); do
            if [ "$CAND" -lt $WEIGHTS_FROM ]; then continue; fi
            if [ $((CAND % WEIGHTS_INTERVAL)) -ne 0 ]; then continue; fi
            if [ "$CAND" -gt "$STEP" ]; then break; fi
            if grep -q "^$CAND$" $PUSHED; then continue; fi
            convert_and_push_weights $CAND
        done
    fi

    for M in "${FULL_MILESTONES[@]}"; do
        if [ "$STEP" -ge "$M" ] && ! grep -q "^$M$" $PUSHED_FULL; then
            if [ -d $WS/outputs/sft/step$M ]; then
                push_full_ckpt $M
            fi
        fi
    done

    cleanup_local_ckpts

    DISK_PCT=$(df / | tail -1 | awk "{print \$5}" | tr -d %)
    if [ "$DISK_PCT" -gt 85 ]; then
        notify "[OneRec][WARN] disk ${DISK_PCT}% used, force-cleaning all non-recent non-milestone"
        cleanup_local_ckpts
    fi

    if [ "$STEP" -ge $((LAST_LOG_STEP + 500)) ] && [ "$STEP" -gt 0 ]; then
        cd $WS/backup
        mkdir -p logs scripts
        cp $WS/logs/*.log logs/ 2>/dev/null || true
        cp $WS/code/OpenOneRec/pretrain/examples/posttrain_sft_1gpu.sh scripts/ 2>/dev/null || true
        cp $WS/run_full_train.sh scripts/ 2>/dev/null || true
        cp $WS/backup_loop.sh scripts/ 2>/dev/null || true
        git add -A
        git -c user.name=dqding -c user.email=dingdraqian@hotmail.com commit -qm "logs at step $STEP" 2>/dev/null && \
            git push -q origin main 2>&1 | tail -3 || true
        LAST_LOG_STEP=$STEP
    fi
done
