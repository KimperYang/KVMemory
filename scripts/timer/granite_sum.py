"""
Prefill latency micro-benchmark for Granite kvlink (sum-attention) models.

Simulates the inference path where document KVs have been pre-cached offline:

1. Run one forward to synthesize `batch_size` per-chunk KV caches.
2. Move the cache to CPU, then back to GPU (measures host<->device transfer).
3. Concat the per-chunk KVs along the sequence axis into a single cache.
4. Re-apply RoPE with global position ids (each chunk was originally rotated
   from position 0).
5. Run the "link prefill": `[sys | <mem_start> | link tokens | <mem_end> | user]`
   against the concatenated cache, with a kvlink-style 4-D attention mask.

Only step 2-5 are timed; step 1 is setup.

```
python scripts/timer/granite_sum.py \
    --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
    --reencode_num 5 --batch_size 10 --sequence_length 500
```
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def move_cache_device(kv, device):
    for layer in range(len(kv)):
        kv[layer][0].to(device)
        kv[layer][1].to(device)


def create_position_ids(batch_size: int, seq_length: int, start_pos: int, reencode_num: int):
    cache_position_ids = []
    sum_position_ids = []
    for i in range(batch_size):
        item_start = start_pos + i * (seq_length + reencode_num)
        cache_position_ids += list(range(item_start, item_start + seq_length))
        sum_position_ids += list(range(item_start + seq_length, item_start + seq_length + reencode_num))
    return cache_position_ids, sum_position_ids


def concat_past_key_values(past_key_values: tuple):
    new_past_key_values = []
    for (keys, values) in past_key_values:
        B, n_heads, S, D = keys.shape
        keys_cat = keys.permute(1, 0, 2, 3).reshape(n_heads, B * S, D).unsqueeze(0)
        values_cat = values.permute(1, 0, 2, 3).reshape(n_heads, B * S, D).unsqueeze(0)
        new_past_key_values.append((keys_cat, values_cat))
    return tuple(new_past_key_values)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (rotate_half(k) * sin)


def apply_rotary_to_concat_kv(concat_past_key_values, cos, sin):
    new_past_key_values = []
    for (key_states, value_states) in concat_past_key_values:
        k_rot = apply_rotary_pos_emb(key_states, cos, sin)
        new_past_key_values.append((k_rot, value_states))
    return tuple(new_past_key_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="HF-format Granite kvlink checkpoint (e.g. training_res/kvlink_5_granite_8B/checkpoint-6000).")
    parser.add_argument("--reencode_num", type=int, default=5,
                        help="Number of link tokens per chunk. Must match the training config.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of pre-cached document chunks.")
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

    # Special tokens added during training. These must already be present in the
    # tokenizer that ships with the checkpoint.
    mem_start = tokenizer.convert_tokens_to_ids("<mem_start>")
    mem_end = tokenizer.convert_tokens_to_ids("<mem_end>")
    if args.reencode_num > 0:
        link0_id = tokenizer.convert_tokens_to_ids("<link_0>")
        if link0_id is None or link0_id == tokenizer.unk_token_id:
            raise ValueError("Checkpoint tokenizer is missing <link_0>; is this really a kvlink5 ckpt?")
    else:
        link0_id = 0  # unused when reencode_num == 0

    # Use the model's own rotary embedding module (architecture-agnostic).
    rotary = model.model.rotary_emb

    # Granite base content vocab 0..100255; specials start at 100256.
    vocab_size = 100256
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    reencode_num = args.reencode_num
    total_time = 0.0

    for i in range(args.iters):
        sys_ids = np.random.randint(0, vocab_size, size=10).tolist()
        user_ids = np.random.randint(0, vocab_size, size=10).tolist()
        sum_ids = [link0_id] * reencode_num

        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, sequence_length),
            device=model.device,
        )

        with torch.no_grad():
            out = model(input_ids)
            old_kv = out.past_key_values

        move_cache_device(old_kv, "cpu")

        torch.cuda.synchronize()
        start_time = time.time()
        move_cache_device(old_kv, model.device)

        concat_kv = concat_past_key_values(old_kv)
        cache_position_ids, sum_position_ids = create_position_ids(
            batch_size=batch_size,
            seq_length=sequence_length,
            start_pos=len(sys_ids),
            reencode_num=reencode_num,
        )
        cos, sin = rotary(
            x=concat_kv[0][0].to(dtype=torch.bfloat16),
            position_ids=torch.tensor([cache_position_ids], device=model.device),
        )
        new_kv = apply_rotary_to_concat_kv(concat_kv, cos=cos, sin=sin)

        generate_id = sys_ids + [mem_start] + sum_ids * batch_size + [mem_end] + user_ids
        sum_position_ids = (
            list(range(0, len(sys_ids) + 1))
            + sum_position_ids
            + list(range(sum_position_ids[-1] + 1, sum_position_ids[-1] + 1 + len(user_ids) + 1))
        )

        input_matrix = torch.triu(
            torch.full(
                (len(generate_id), len(generate_id)),
                float("-inf"),
                dtype=torch.bfloat16,
                device=model.device,
            ),
            diagonal=1,
        )
        cache_matrix = torch.full(
            (len(generate_id), batch_size * sequence_length),
            float("-inf"),
            dtype=torch.bfloat16,
            device=model.device,
        )
        for idx in range(batch_size):
            cache_matrix[
                len(sys_ids) + 1 : len(sys_ids) + 1 + reencode_num,
                : idx * sequence_length,
            ] = float(0)

        attention_matrix = torch.cat([cache_matrix, input_matrix], dim=1).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            model(
                input_ids=torch.tensor([generate_id], device=model.device),
                past_key_values=new_kv,
                attention_mask=attention_matrix,
                position_ids=torch.tensor([sum_position_ids], device=model.device),
                cache_position=torch.tensor([cache_position_ids], device=model.device),
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
