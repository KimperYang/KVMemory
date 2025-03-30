#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 7 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 8 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 9 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 7 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 8 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 --pos 9 &


wait
