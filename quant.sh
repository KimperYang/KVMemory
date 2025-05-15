CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress_block/lingua.py --ckpt 1122 --run "training_res/block_1B_qa" --pos 9 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress_block/lingua_hqa.py --ckpt 1122 --run "training_res/block_1B_qa" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress_block/lingua_musique.py --ckpt 1122 --run "training_res/block_1B_qa" &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress_block/lingua_tqa.py --ckpt 1122 --run "training_res/block_1B_qa" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress_block/lingua_wiki.py --ckpt 1122 --run "training_res/block_1B_qa" &