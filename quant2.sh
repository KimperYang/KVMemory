CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/quant_block/tqa_block.py --ckpt 1122 --run "training_res/block_1B_qa"  &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/quant_block/wiki_block.py --ckpt 1122 --run "training_res/block_1B_qa"  &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/quant_block/musique_block.py --ckpt 1122 --run "training_res/block_1B_qa"  &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/quant_block/hqa_block.py --ckpt 1122 --run "training_res/block_1B_qa"  &