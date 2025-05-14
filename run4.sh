CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --run "training_res/qa/8B/sum_5_31_8B_qa" --reencode 5 --ckpt 1122 --pos 6&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --run "training_res/qa/8B/sum_5_31_8B_qa" --reencode 5 --ckpt 1122 --pos 7&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --run "training_res/qa/8B/sum_5_31_8B_qa" --reencode 5 --ckpt 1122 --pos 8&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --run "training_res/qa/8B/sum_5_31_8B_qa" --reencode 5 --ckpt 1122 --pos 9&

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_block.py --run "training_res/qa/8B/block_31_8B_qa"  --ckpt 1122 --pos 6&
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_block.py --run "training_res/qa/8B/block_31_8B_qa"  --ckpt 1122 --pos 7&
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_block.py --run "training_res/qa/8B/block_31_8B_qa"  --ckpt 1122 --pos 8&
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_block.py --run "training_res/qa/8B/block_31_8B_qa"  --ckpt 1122 --pos 9&