#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 8 --run "new_data/seq" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 1 --run "new_data/seq" &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 2 --run "new_data/seq" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 3 --run "new_data/seq" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 4 --run "new_data/seq" &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 5 --run "new_data/seq" &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 6 --run "new_data/seq" &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 7 --run "new_data/seq" &

wait