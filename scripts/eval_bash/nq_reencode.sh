#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_new_mix_bsz64" --ckpt 6000 --pos 1 --reencode 1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_new_mix_bsz64" --ckpt 6000 --pos 2 --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_new_mix_bsz64" --ckpt 6000 --pos 3 --reencode 1 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_1_new_mix_bsz64" --ckpt 6000 --pos 4 --reencode 1 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_0_new_mix_bsz64" --ckpt 6000 --pos 1 --reencode 0 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_0_new_mix_bsz64" --ckpt 6000 --pos 2 --reencode 0 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_0_new_mix_bsz64" --ckpt 6000 --pos 3 --reencode 0 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_sum.py --run "sum/sum_0_new_mix_bsz64" --ckpt 6000 --pos 4 --reencode 0 &

wait