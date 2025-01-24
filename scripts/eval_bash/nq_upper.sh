#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 9 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 7 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 4 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 5 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_new_mix" --ckpt 6000 --pos 7 &

wait
