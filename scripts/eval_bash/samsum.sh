#!/bin/bash

source /home/jingbo/.bashrc
conda activate unlearning
cd /home/jingbo/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_sum.py --run "sum/sum_0_prompt" --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_sum.py --run "sum/sum_1_prompt" --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/samsum/samsum_sum.py --run "sum/sum_5_prompt" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/samsum/samsum_block.py --run "new_data/block_prompt" &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/samsum/samsum_upper.py --run "new_data/upper_prompt" &

wait