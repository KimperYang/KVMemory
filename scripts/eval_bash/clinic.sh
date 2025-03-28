#!/bin/bash

# source /home/jingbo/.bashrc
conda activate unlearning
cd /home/jingbo/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/multinews/multinews_sum.py --run "sum/sum_0_prompt" --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/multinews/multinews_sum.py --run "sum/sum_1_prompt" --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/multinews/multinews_sum.py --run "sum/sum_5_prompt"  --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/multinews/multinews_block.py --run "new_data/block_prompt" &

wait
