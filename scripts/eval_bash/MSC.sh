#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate unlearning
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/MSC/dialog_sum.py --run "sum/sum_1_prompt" --ckpt 6000 --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/MSC/dialog_sum.py --run "sum/sum_5_prompt" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/MSC/dialog_sum.py --run "sum/sum_0_prompt" --ckpt 6000 --reencode 0 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/MSC/dialog_upper.py --run "new_data/upper_prompt" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/MSC/dialog_upper.py --run "meta-llama/Llama-3.2-1B-Instruct" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/MSC/dialog_block.py --run "meta-llama/Llama-3.2-1B-Instruct" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/MSC/dialog_block.py --run "new_data/block_prompt" --ckpt 6000 &

wait