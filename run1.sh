# startTime=$(date +%s)

# output_dir=$1
# run_name=$2

# export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/hqa/hqa_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/tqa/tqa_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 3 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 4 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 5 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 6 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 7 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 8 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_0_31_8B" --reencode 0 --pos 9 &