#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_sum.py --run "sum/sum_1_new_mix_bsz64" --ckpt 6000 --reencode 1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_sum.py --run "sum/sum_0_new_mix_bsz64" --ckpt 6000 --reencode 0 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_sum.py --run "sum/sum_5_new_mix" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_upper.py --run "new_data/upper_new_mix" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_block.py --run "new_data/block_new_mix" --ckpt 6000 &

wait