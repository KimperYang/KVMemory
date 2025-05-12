startTime=$(date +%s)

output_dir=$1
run_name=$2

export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_blend.py --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_blend.py --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_blend.py --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_blend.py --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_blend.py --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_blend.py --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_blend.py --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_blend.py --pos 7 &