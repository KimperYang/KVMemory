
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 0 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 1 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 2 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 3 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 4 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 5 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 6 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 7 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 8 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 --pos 9 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/tqa/tqa_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/musique/musique_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/hqa/hqa_sum.py --ckpt 1122 --run "training_res/qa/3B/sum_0_3B_qa" --reencode 0 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 0 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 1 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 2 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 3 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 4 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 5 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 6 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 7 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 8 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" --pos 9 &

wait

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/tqa/tqa_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/2wiki/wiki_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/musique/musique_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/hqa/hqa_upper.py --ckpt 1122 --run "training_res/qa/3B/upper_3B_qa" &