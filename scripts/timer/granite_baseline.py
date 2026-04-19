"""
Prefill latency micro-benchmark for Granite (upperbound / standard attention).

Single forward pass over `[sys_ids | batch_size * sequence_length docs | user_ids]`
as a flat sequence, no memory-token tricks.

```
python scripts/timer/granite_baseline.py \
    --ckpt_path training_res/upperbound_granite_8B/checkpoint-6000 \
    --batch_size 10 --sequence_length 500
```
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="HF-format Granite checkpoint (upperbound or kvlink).")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of document chunks concatenated into the prefill.")
    parser.add_argument("--sequence_length", type=int, default=500,
                        help="Tokens per document chunk.")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warm-up iterations excluded from the average.")
    parser.add_argument("--iters", type=int, default=110,
                        help="Total iterations; average is over (iters - warmup).")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    # Granite base vocab occupies 0..100255; specials (pad/eot/role/link/mem) start at 100256.
    # Sample below that to keep the synthetic prompt free of special tokens.
    vocab_size = 100256
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    total_time = 0.0

    for i in range(args.iters):
        sys_ids = np.random.randint(0, vocab_size, size=10).tolist()
        user_ids = np.random.randint(0, vocab_size, size=10).tolist()
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, sequence_length),
            device=model.device,
        )

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            model(
                input_ids=torch.tensor(
                    [sys_ids + list(input_ids.view(-1).cpu().numpy()) + user_ids],
                    device=model.device,
                ),
                use_cache=True,
            )
        torch.cuda.synchronize()
        end_time = time.time()

        print("time:", end_time - start_time)
        if i >= args.warmup:
            total_time += end_time - start_time

    print("Avg time:", total_time / (args.iters - args.warmup))


if __name__ == "__main__":
    main()
