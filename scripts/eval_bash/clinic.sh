#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/hqa/hqa_sum.py --run "sum/sum_5_remove_sftmem" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/tqa/tqa_sum.py --run "sum/sum_5_remove_sftmem" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_sum.py --run "sum/sum_5_remove_sftmem" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_sum.py --run "sum/sum_5_remove_sftmem" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/hqa/hqa_sum.py --run "sum/sum_5_remove_sum" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/tqa/tqa_sum.py --run "sum/sum_5_remove_sum" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/musique/musique_sum.py --run "sum/sum_5_remove_sum" --ckpt 6000 --reencode 5 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/2wiki/wiki_sum.py --run "sum/sum_5_remove_sum" --ckpt 6000 --reencode 5 &

wait