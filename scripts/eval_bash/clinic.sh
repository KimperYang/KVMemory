#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/icl/clinic_sum.py --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/icl/clinic_sum.py --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/icl/clinic_sum.py --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/icl/clinic_block.py --path "training_res/new_data/block_new_mix/checkpoint-6000" &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/icl/clinic_block.py --path "/dccstor/scllm/Block-Attention/training_res/checkpoint-624" &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/icl/clinic_block.py --path "meta-llama/Llama-3.2-1B-Instruct" &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/icl/clinic_upper.py --path "training_res/new_data/upper_new_mix/checkpoint-6000" &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/icl/clinic_upper.py --path "meta-llama/Llama-3.2-1B-Instruct" &

wait
