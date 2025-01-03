#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_reencode.py --ckpt 10000 --run "reencode_5_bsz256" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_reencode.py --ckpt 8000 --run "reencode_10_bsz256" --reencode 10 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_seq.py --ckpt 10000 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_block.py --ckpt "/dccstor/scllm/Block-Attention/training_res/checkpoint-624" &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_promptcache.py &

wait