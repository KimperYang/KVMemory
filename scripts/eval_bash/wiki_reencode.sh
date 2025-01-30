#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "sum/sum_0_prompt" --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "sum/sum_1_prompt" --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "sum/sum_5_prompt" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_block.py --ckpt 6000 --run "new_data/block_prompt" &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_upper.py --ckpt 6000 --run "new_data/upper_prompt" &

wait