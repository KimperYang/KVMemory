# startTime=$(date +%s)

# output_dir=$1
# run_name=$2

# export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"


CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 --pos 9 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/hqa/hqa_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/tqa/tqa_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 6000 --run "training_res/sum/sum_5_1B_qa" --reencode 5 &