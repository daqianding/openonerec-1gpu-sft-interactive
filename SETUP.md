# Resuming OneRec SFT Training on a Fresh VM

This guide gets a new A100 80GB VM from zero to **resuming SFT training from the latest HF checkpoint** in ~1.5 hours.

## 0. Prerequisites
- Single A100 80GB (or H100 80GB) Linux VM, NVIDIA driver ≥ 535, ~1 TB free disk on `/`
- Sudo access for Docker
- Tokens:
  - **HF token** with read+write to `dqding/OneRec-1.7B-sft-interactive-1gpu` (private datasets read access for raw data is gated; request access if needed)
  - **Optional**: GitHub PAT for log mirroring

---

## 1. Install Docker + NVIDIA Container Toolkit
```bash
# Docker (skip if installed)
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
# NVIDIA toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi  # verify
```

## 2. Install hf CLI + login
```bash
pip install --user 'huggingface_hub[cli]'
export PATH=$HOME/.local/bin:$PATH
echo 'hf_YOUR_TOKEN' > ~/.hf_token && chmod 600 ~/.hf_token
hf auth login --token $(cat ~/.hf_token)
```

## 3. Clone repos and code
```bash
mkdir -p ~/workspace/code ~/workspace/data ~/workspace/docker ~/workspace/logs ~/workspace/outputs
cd ~/workspace/code
# OpenOneRec source
git clone https://github.com/Kuaishou-OneRec/OpenOneRec.git
# Mirror of our 1-GPU patches + scripts (this repo)
cd ~/workspace
git clone https://github.com/daqianding/openonerec-1gpu-sft-interactive.git backup
```

## 4. Apply our 1-GPU patches to OpenOneRec
```bash
# Patched train_qwen3.py — adds num_training_steps break + DTensor cross-mesh rebind
cp ~/workspace/backup/scripts/train_qwen3.py.patched \
   ~/workspace/code/OpenOneRec/pretrain/recipes/train_qwen3.py
# Single-GPU launcher (no mpirun, world_size=1)
cp ~/workspace/backup/scripts/posttrain_sft_1gpu.sh \
   ~/workspace/code/OpenOneRec/pretrain/examples/posttrain_sft_1gpu.sh
chmod +x ~/workspace/code/OpenOneRec/pretrain/examples/posttrain_sft_1gpu.sh
# Resume launcher (root of workspace)
cp ~/workspace/backup/scripts/run_full_train_resume.sh ~/workspace/run_full_train_resume.sh
chmod +x ~/workspace/run_full_train_resume.sh
# Dockerfile
cp ~/workspace/backup/docker/Dockerfile.train ~/workspace/docker/Dockerfile.train
```

## 5. Build training Docker image (~10 min)
```bash
cd ~/workspace/docker
sudo docker build -t onerec-train:latest -f Dockerfile.train . 2>&1 | tee ~/workspace/logs/docker_build.log
# Verify
sudo docker run --rm onerec-train:latest python3 -c "import torch; print(torch.__version__)"  # 2.5.0a0+...
```

## 6. Download base model (~10 min, 4 GB)
```bash
hf download Kuaishou-OneRec/OneRec-1.7B-pretrain \
  --local-dir ~/workspace/data/hf_models/OneRec-1.7B-pretrain
```

## 7. Download dataset (~30 min, 8.4 GB raw → 1.6 GB sharded)
```bash
# Raw RecIF data — gated, request access first
hf download Kuaishou-OneRec/OpenOneRec-RecIF \
  --repo-type dataset \
  --local-dir ~/workspace/data/raw_data/onerec_data
```

### Preprocess: interactive_rec only
```bash
cd ~/workspace/code/OpenOneRec/data/onerec_data
# Disable all RUN_* except interactive_rec
sed -i 's|^RUN_[A-Z_]*=1|&_DISABLED|' run.sh
sed -i 's|^\(RUN_[A-Z_]*\)_DISABLED=1$|\1=0|' run.sh
sed -i 's|^RUN_SFT_INTERACTIVE_REC=0|RUN_SFT_INTERACTIVE_REC=1|' run.sh
sed -i 's|../../raw_data/onerec_data|/home/'$USER'/workspace/data/raw_data/onerec_data|g' run.sh
sed -i 's|../../output|/home/'$USER'/workspace/data/output|g' run.sh
mkdir -p ~/workspace/data/output
sudo docker run --rm -v ~/workspace:/home/$USER/workspace -w /home/$USER/workspace/code/OpenOneRec/data/onerec_data \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  bash -c 'pip install -q pandas numpy tqdm pyarrow && bash run.sh'

# Shard into split_data_sft/
cd ~/workspace/code/OpenOneRec/data
sed -i 's|^GENERAL_TEXT_PATH=.*|GENERAL_TEXT_PATH=""|' prepare_sft.sh
sed -i 's|^REC_DATA_PATH=.*|REC_DATA_PATH="/home/'$USER'/workspace/data/output"|' prepare_sft.sh
sed -i 's|^OUTPUT_DIR=.*|OUTPUT_DIR="/home/'$USER'/workspace/data/split_data_sft"|' prepare_sft.sh
sudo docker run --rm -v ~/workspace:/home/$USER/workspace -w /home/$USER/workspace/code/OpenOneRec/data \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  bash -c 'pip install -q pandas numpy tqdm pyarrow && bash prepare_sft.sh'
ls ~/workspace/data/split_data_sft/  # should have ~426 parquet files + file_list.json
```

## 8. Pull latest checkpoint from HF (~10 min, 22 GB)
```bash
# Replace stepN with the latest ckpt available on HF (e.g. step30000, step40000)
LATEST_STEP=30000
mkdir -p ~/workspace/outputs/sft
hf download dqding/OneRec-1.7B-sft-interactive-1gpu \
  --include "ckpt/step${LATEST_STEP}/*" \
  --local-dir /tmp/hf_dl
# distcp + optimizer + dataloader 三件套
mv /tmp/hf_dl/ckpt/step${LATEST_STEP} ~/workspace/outputs/sft/step${LATEST_STEP}
echo step${LATEST_STEP} > ~/workspace/outputs/sft/latest
ls ~/workspace/outputs/sft/step${LATEST_STEP}/
# Should show: dataloader_ckpt/  global_step${LATEST_STEP}/  optimizer_ckpt/
```

## 9. Resume training
```bash
# Edit NUM_STEPS in run_full_train_resume.sh if you want a different target (default 40000)
# Then launch:
bash ~/workspace/run_full_train_resume.sh ${LATEST_STEP}
```

You should see in the log:
```
Resume from checkpoint: ...step30000/global_step30000, global_step=30000
Successfully loaded model using distributed checkpoint
[resume] rebound 620 optimizer DTensor states to live device mesh
Successfully loaded optimizer and scheduler state
Successfully loaded dataloader state
```
Then steps tick at ~4.5 s/step on A100 80GB.

## 10. (Optional) Re-enable backup loop to mirror new ckpts to HF
```bash
cp ~/workspace/backup/scripts/backup_loop.sh ~/workspace/backup_loop.sh
# Inside tmux:
tmux new -s backup
bash ~/workspace/backup_loop.sh 2>&1 | tee -a ~/workspace/logs/backup_loop.log
```

## Key gotchas
- **Make HF repo public** if you hit "Private repository storage limit reached" (private free tier appears to throttle around 30 GB despite docs).
- **Cleanup local ckpts**: `backup_loop.sh` keeps only last 2 + milestones — without it 22 GB/ckpt × every 500 steps will fill 1 TB in ~10 hours.
- **DTensor cross-mesh on resume**: fixed via `_rebind_dtensor_to_mesh` in train_qwen3.py — without this patch, the first `optimizer.step()` after resume crashes with `aten._foreach_lerp_.Scalar: DTensor does not support cross-mesh operation yet`.
- **num_training_steps break**: stock train_qwen3.py never exits at NUM_STEPS — patched line ~1238 to break.
- **Dataloader resume**: `StatefulDataLoader` state file pins to absolute parquet paths under `/home/v-daqianding/workspace/data/split_data_sft/` — keep the same username path or symlink `/home/v-daqianding` → `$HOME`. (Or accept dataloader will reset to position 0; loss will rise briefly then recover.)

## Training params reference
- MODEL: Qwen3-1.7B-pretrain (OneRec)
- MAX_LENGTH=32768, MINIBATCH=16384
- LR=2e-4 → MIN_LR=1e-4 cosine, NUM_WARMUP=3000
- NUM_STEPS=40000, SAVE_PER_STEP=500
- bf16 weights via fp32 master, gradient checkpointing on, chunked loss, FSDP world_size=1

---

## ✅ Last Resume Point: step38500 (2026-04-27 03:00 UTC)

Training stopped at **step38500 / 40000** to free GPU for interactive benchmark before Azure VM rental ended.

**HF state at handoff**:
- `ckpt/step38500` — full distcp + optimizer + dataloader (USE THIS to resume)
- `ckpt/step30000`, `ckpt/step15000` — earlier full milestones
- `weights/step38000` — latest bf16 safetensors (use for benchmark/inference)

**To resume on a new VM**:
```bash
LATEST_STEP=38500
hf download dqding/OneRec-1.7B-sft-interactive-1gpu \
  --include "ckpt/step${LATEST_STEP}/*" --local-dir /tmp/hf_dl
mkdir -p ~/workspace/outputs/sft
mv /tmp/hf_dl/ckpt/step${LATEST_STEP} ~/workspace/outputs/sft/step${LATEST_STEP}
echo step${LATEST_STEP} > ~/workspace/outputs/sft/latest
bash ~/workspace/run_full_train_resume.sh ${LATEST_STEP}
# This will resume from step38500 and train to step40000 (1500 more steps, ~1.85h on A100)
```
