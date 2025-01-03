#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_promptcache.py --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_promptcache.py --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_promptcache.py --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_promptcache.py --pos 3 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_promptcache.py --pos 4 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_promptcache.py --pos 5 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_promptcache.py --pos 6 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_promptcache.py --pos 7 &

wait
