#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 9 --reencode 5 
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 3 --reencode 5 
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 4 --reencode 5 
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 5 --reencode 5 
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 4 --reencode 5 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 5 --reencode 5 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 6 --reencode 5 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --pos 7 --reencode 5 &

wait