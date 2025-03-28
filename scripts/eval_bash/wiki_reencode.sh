#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_upper.py --ckpt 6000 --run "new_data/upper_3B" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_original.py &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_promptcache.py &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_block.py --ckpt 6000 --run "new_data/block_3B" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_upper.py --ckpt 6000 --run "new_data/upper_3B" &

wait