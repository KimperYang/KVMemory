#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/MSC/dialog_bias.py --run "new_data/bias" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/MSC/dialog_reencode.py --run "new_data/reencode_1" --ckpt 6000 --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/MSC/dialog_reencode.py --run "new_data/reencode_5" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/MSC/dialog_upper.py --run "new_data/upper" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/MSC/dialog_upper.py --run "meta-llama/Llama-3.2-1B-Instruct" --ckpt 0 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/MSC/dialog_seq.py --run "new_data/seq" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/MSC/dialog_block.py --run "meta-llama/Llama-3.2-1B-Instruct" --ckpt 0 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/MSC/dialog_block.py --run "new_data/block" --ckpt 6000 &

wait
