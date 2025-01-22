#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_compress.py --ckpt 6000 --run "compress/compress_20" --reencode 20 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_compress.py --ckpt 6000 --run "compress/compress_50" --reencode 50 &

wait