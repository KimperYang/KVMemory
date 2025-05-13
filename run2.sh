# startTime=$(date +%s)

# output_dir=$1
# run_name=$2

# export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/hqa/hqa_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/tqa/tqa_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/musique/musique_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" &

# wait

# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 0 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 1 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 2 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 3 &

# wait

# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 4 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 5 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 6 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 7 &

# wait

# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 8 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 9 &


# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 0 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 1 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 2 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 3 &

# wait

# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 4 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 5 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 6 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 7 &

# wait

# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 8 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa" --pos 9 &

# wait

# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/hqa/hqa_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa"  &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/tqa/tqa_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa"  &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/musique/musique_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa"  &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/2wiki/wiki_block.py --ckpt 1122 --run "training_res/new_data/block_1B_qa"  &

python scripts/evaluation/compress/lingua_hqa.py --run "training_res/sum/sum_5_1B_qa" --reencode 5 --ckpt 1122
python scripts/evaluation/compress_block/lingua_hqa.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122