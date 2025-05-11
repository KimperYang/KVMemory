#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 0&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 1&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 0&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 1&
wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 2&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 3&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 2&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 3&
wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 4&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 5&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 4&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 5&
wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 6&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 7&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 6&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 7&
wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 8&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_5_31_8B_qa" --reencode 5 --pos 9&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 8&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa" --pos 9&