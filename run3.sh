CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 0 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 1 &

wait

CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 2 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 3 &

wait

CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 4 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 5 &

wait

CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 6 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 7 &

wait

CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 8 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_block.py --ckpt 1122 --run "training_res/qa/3B/block_3B_qa" --pos 9 &

wait