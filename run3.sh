startTime=$(date +%s)

output_dir=$1
run_name=$2

export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/hqa/hqa_blend.py &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/2wiki/wiki_blend.py &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/musique/musique_blend.py &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/tqa/tqa_blend.py &

# wait

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_blend.py --pos 4 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_blend.py --pos 5 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_blend.py --pos 6 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_blend.py --pos 7 &

# wait

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_blend.py --pos 8 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_blend.py --pos 9 &