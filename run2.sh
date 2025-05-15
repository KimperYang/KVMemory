CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_sum.py  --run "training_res/sum_0_31_8B" --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_sum.py  --run "training_res/sum_1_31_8B" --reencode 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/samsum/samsum_sum.py  --run "training_res/sum_5_31_8B" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/samsum/samsum_block.py  --run "block_31_8B" &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/samsum/samsum_upper.py  --run "meta-llama/Llama-3.1-8B-Instruct" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/samsum/samsum_blend.py &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/samsum/samsum_block.py  --run "meta-llama/Llama-3.1-8B-Instruct" &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py  --run "training_res/sum_5_31_8B" --pos 7 &