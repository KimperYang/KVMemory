startTime=$(date +%s)

output_dir=$1
run_name=$2

export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

python scripts/evaluation/tqa/tqa_block.py --ckpt 1122 --run "training_res/new_data/block_31_8B_qa"