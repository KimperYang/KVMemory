CUDA_VISIBLE_DEVICES=2

python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 0
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 1
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 2
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 3
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 4
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 5
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 6
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 7
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 8
python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum_0_31_8B" --reencode 0 --pos 9

python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 0
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 1
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 2
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 3
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 4
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 5
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 6
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 7
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 8
python scripts/evaluation/nq/nq_block.py --ckpt 6000 --run "training_res/block_31_8B" --pos 9