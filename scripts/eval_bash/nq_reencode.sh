#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p" --ckpt 6000 --pos 5 --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p" --ckpt 6000 --pos 6 --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p" --ckpt 6000 --pos 7 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p" --ckpt 6000 --pos 8 --reencode 5 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p_8k" --ckpt 8000 --pos 5 --reencode 5 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p_8k" --ckpt 8000 --pos 6 --reencode 5 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p_8k" --ckpt 8000 --pos 7 --reencode 5 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_5_3B_10p_8k" --ckpt 8000 --pos 8 --reencode 5 &

wait