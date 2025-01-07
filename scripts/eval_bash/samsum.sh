#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_bias.py --ckpt 6000 --run "new_data/bias" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_upper.py --ckpt 6000 --run "new_data/baseline" &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/samsum/samsum_seq.py --ckpt 6000 --run "new_data/seq" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/samsum/samsum_reencode.py --ckpt 6000 --reencode 5 --run "new_data/reencode_5" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/samsum/samsum_reencode.py --ckpt 6000 --pos 4 --reencode 5 --run "new_data/reencode_5" &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/samsum/samsum_reencode.py --ckpt 6000 --pos 5 --reencode 5 --run "new_data/reencode_5" &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/samsum/samsum_reencode.py --ckpt 6000 --pos 6 --reencode 5 --run "new_data/reencode_5" &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/samsum/samsum_reencode.py --ckpt 6000 --pos 7 --reencode 5 --run "new_data/reencode_5" &

wait