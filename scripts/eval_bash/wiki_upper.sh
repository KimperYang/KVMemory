#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_upper.py --run "new_data/baseline_2e-5" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 &

wait