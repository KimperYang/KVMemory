#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/tqa/tqa_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --reencode 5 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/hqa/hqa_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --reencode 5 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_sum.py --run "/mnt/data2/jingbo/sum_5_8B" --ckpt 6000 --reencode 5 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/tqa/tqa_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/musique/musique_block.py --run "/mnt/data2/jingbo/block_8B" --ckpt 6000 &

wait