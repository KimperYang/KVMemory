#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate unlearning
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/musique/musique_upper.py --run "new_data/upper_prompt" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_upper.py --run "new_data/upper_prompt" --ckpt 6000 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/hqa/hqa_upper.py --run "new_data/upper_prompt" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/musique/musique_block.py --run "block_model_1B" --ckpt 624 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_block.py --run "block_model_1B" --ckpt 624 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/tqa/tqa_block.py --run "block_model_1B" --ckpt 624 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_blend.py --run "meta-llama/Llama-3.2-3B-Instruct" --weight 3 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_blend.py --run "meta-llama/Llama-3.2-1B-Instruct" --weight 1 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_blend.py --run "meta-llama/Llama-3.2-3B-Instruct" --weight 3 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/hqa/hqa_blend.py --run "meta-llama/Llama-3.2-1B-Instruct" --weight 1 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/musique/musique_blend.py --run "meta-llama/Llama-3.2-3B-Instruct" --weight 3 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/musique/musique_blend.py --run "meta-llama/Llama-3.2-1B-Instruct" --weight 1 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/musique/musique_sum.py --run "sum/sum_0_prompt" --ckpt 6000 --reencode 0 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_sum.py --run "sum/sum_1_prompt" --ckpt 6000 --reencode 1 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/musique/musique_sum.py --run "sum/sum_5_prompt" --ckpt 6000 --reencode 5 &

wait