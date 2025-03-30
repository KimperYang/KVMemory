#!/bin/bash

# source /u/shiyuucsb/.bashrc
conda activate unlearning
# cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 3 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 4 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 5 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 6 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_upper.py --run "new_data/upper_8B" --ckpt 6000 --pos 7 &

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_upper.py --run "new_data/upper_8B" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_upper.py --run "new_data/upper_8B" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_upper.py --run "new_data/upper_8B" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/tqa/tqa_upper.py --run "new_data/upper_8B" --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_upper.py --run ""meta-llama/Meta-Llama-3-8B-Instruct"" --ckpt 0 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/hqa/hqa_upper.py --run ""meta-llama/Meta-Llama-3-8B-Instruct"" --ckpt 0 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/musique/musique_upper.py --run ""meta-llama/Meta-Llama-3-8B-Instruct"" --ckpt 0 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/tqa/tqa_upper.py --run ""meta-llama/Meta-Llama-3-8B-Instruct"" --ckpt 0 &

wait