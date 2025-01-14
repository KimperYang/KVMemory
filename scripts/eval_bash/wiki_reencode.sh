#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "sum/sum_1" --reencode 1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "sum/sum_2" --reencode 2 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_seq.py --run "new_data/seq" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_bias.py --run "new_data/bias" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_reencode.py --ckpt 6000 --run "new_data/reencode_1" --reencode 1 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_block.py --ckpt "/dccstor/scllm/Block-Attention/training_res/checkpoint-624" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_promptcache.py &

wait