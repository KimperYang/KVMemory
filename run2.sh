CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_31_8B" --pos 9 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_upper.py --ckpt 1122 --run "upper_31_8B" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_upper.py --ckpt 1122 --run "upper_31_8B" &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_upper.py --ckpt 1122 --run "upper_31_8B" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/hqa/hqa_upper.py --ckpt 1122 --run "upper_31_8B" &