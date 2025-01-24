#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 8 --reencode 1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 9 --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 2 --reencode 1 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 3 --reencode 1 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 4 --reencode 1 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 5 --reencode 1 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 6 --reencode 1 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_shuffle" --ckpt 6000 --pos 7 --reencode 1 &

wait