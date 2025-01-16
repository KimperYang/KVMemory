#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 0 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 1 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 8 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 9 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 4 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 5 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 --pos 7 &

wait
