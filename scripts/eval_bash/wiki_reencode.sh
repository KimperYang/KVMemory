#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "training_res/sum/sum_5_31_8B" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_sum.py --ckpt 6000 --run "training_res/sum/sum_5_31_8B" --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_sum.py --ckpt 6000 --run "training_res/sum/sum_5_31_8B" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/tqa/tqa_sum.py --ckpt 6000 --run "training_res/sum/sum_5_31_8B" --reencode 5 &

wait