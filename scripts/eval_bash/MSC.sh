#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/MSC/dialog_sum.py --run "sum/sum_1" --ckpt 6000 --reencode 1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/MSC/dialog_sum.py --run "sum/sum_2" --ckpt 6000 --reencode 2 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/MSC/dialog_sum.py --run "sum/sum_0" --ckpt 6000 --reencode 0 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/MSC/dialog_upper.py --run "new_data/upper" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/MSC/dialog_upper.py --run "new_data/baseline_2e-5" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/MSC/dialog_upper.py --run "new_data/baseline_5e-5" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/MSC/dialog_upper.py --run "meta-llama/Llama-3.2-1B-Instruct" --ckpt 6000 &

wait