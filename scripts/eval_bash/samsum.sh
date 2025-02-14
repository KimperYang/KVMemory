#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_upper.py --run "meta-llama/Llama-3.2-3B-Instruct" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_block.py --run "meta-llama/Llama-3.2-3B-Instruct" &

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_sum.py --run "sum/sum_0_3B" --reencode 0 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_sum.py --run "sum/sum_1_3B" --reencode 1 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/samsum/samsum_sum.py --run "sum/sum_5_3B" --reencode 5 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/samsum/samsum_block.py --run "new_data/block_3B" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/samsum/samsum_upper.py --run "new_data/upper_3B" &

wait