#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_upper.py --run "training_res/new_data/upper_new_mix/checkpoint-6000" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_upper.py --run "meta-llama/Llama-3.2-1B-Instruct" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/samsum/samsum_block.py --run "/dccstor/scllm/Block-Attention/training_res/checkpoint-624" &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/samsum/samsum_block.py --run "training_res/new_data/block_new_mix/checkpoint-6000" &

wait