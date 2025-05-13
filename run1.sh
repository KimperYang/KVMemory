# startTime=$(date +%s)

# output_dir=$1
# run_name=$2

# export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 &


# wait

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/musique/musique_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 &

# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 0
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 1
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 2
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 3
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 4
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 5
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 6
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 7
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 8
# python scripts/evaluation/compress/lingua.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122 --pos 9

# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 0
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 1
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 2
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 3
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 4
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 5
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 6
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 7
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 8
# python scripts/evaluation/compress_block/lingua.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122 --pos 9

# python scripts/evaluation/compress/lingua_musique.py --run "training_res/qa/3B/sum_5_3B_qa" --reencode 5 --ckpt 1122
# python scripts/evaluation/compress_block/lingua_musique.py --run "training_res/new_data/block_1B_qa" --reencode 0 --ckpt 1122
