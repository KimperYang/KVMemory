#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_block.py --run "/dccstor/scllm/Block-Attention/training_res" --ckpt 624 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_block.py --run "/dccstor/scllm/KVMemory/training_res/new_data/block" --ckpt 6000 &

wait