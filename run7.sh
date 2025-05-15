CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "upper_8B_qa" --pos 9 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_upper.py --ckpt 1122 --run "upper_8B_qa" &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_upper.py --ckpt 1122 --run "upper_8B_qa" &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_upper.py --ckpt 1122 --run "upper_8B_qa" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/hqa/hqa_upper.py --ckpt 1122 --run "upper_8B_qa" &