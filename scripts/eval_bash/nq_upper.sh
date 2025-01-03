#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 2000 --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 2000 --pos 9 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_promptcache.py --pos 8 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_promptcache.py --pos 9 &

wait
