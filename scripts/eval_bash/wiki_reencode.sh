#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_reencode.py --ckpt 6000 --run "new_data/reencode_5" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_upper.py &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_seq.py --ckpt 6000 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_bias.py --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_block.py --ckpt "/dccstor/scllm/Block-Attention/training_res/checkpoint-624" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_promptcache.py &

wait