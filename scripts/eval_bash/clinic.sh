#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate kvm
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/multinews/multinews_upper.py --run "new_data/upper_3B" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/multinews/multinews_upper.py --run "meta-llama/Llama-3.2-3B-Instruct" --weight 3 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/multinews/multinews_upper.py --run "meta-llama/Llama-3.2-1B-Instruct" --weight 1 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/multinews/multinews_blend.py --weight 3 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/multinews/multinews_blend.py --weight 1 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/multinews/multinews_block.py --run "meta-llama/Llama-3.2-3B-Instruct" --weight 3 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/multinews/multinews_block.py --run "meta-llama/Llama-3.2-1B-Instruct" --weight 1 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/icl/clinic_block.py --path "/dccstor/scllm/Block-Attention/training_res/checkpoint-624" &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/icl/clinic_block.py --path "meta-llama/Llama-3.2-3B-Instruct" &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/icl/clinic_upper.py --path "training_res/new_data/upper_3B/checkpoint-6000" &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/icl/clinic_upper.py --path "meta-llama/Llama-3.2-3B-Instruct" &

wait
