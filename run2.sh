CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/tqa/tqa_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" &


# wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/musique/musique_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/hqa/hqa_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" &