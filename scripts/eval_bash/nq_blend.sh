conda activate kvm

CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_blend.py --pos 0 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_blend.py --pos 1 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_blend.py --pos 2 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_blend.py --pos 3 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_blend.py --pos 4 &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_blend.py --pos 5 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_blend.py --pos 6 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_blend.py --pos 7 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_blend.py --pos 8 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_blend.py --pos 9 &

wait