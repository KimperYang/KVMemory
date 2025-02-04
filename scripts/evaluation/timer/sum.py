import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd    
import json
import time
import string
from typing import List
from peft import PeftModel
import regex
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM
import numpy as np

def create_position_ids(
    batch_size: int,
    seq_length: int,
    start_pos: int,
    reencode_num: int):

    cache_position_ids = []
    sum_position_ids = []

    for i in range(batch_size):
        item_start = start_pos + i * (seq_length + reencode_num)
        cache_position_ids += list(range(item_start, item_start + seq_length))
        sum_position_ids += list(range(item_start + seq_length, item_start + seq_length + reencode_num))
    return cache_position_ids,sum_position_ids

def concat_past_key_values(past_key_values: tuple):

    new_past_key_values = []

    for (keys, values) in past_key_values:
        B, n_heads, S, D = keys.shape

        keys_cat = keys.permute(1, 0, 2, 3).reshape(n_heads, B * S, D).unsqueeze(0)
        values_cat = values.permute(1, 0, 2, 3).reshape(n_heads, B * S, D).unsqueeze(0)

        new_past_key_values.append((keys_cat, values_cat))

    return tuple(new_past_key_values)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def apply_rotary_to_concat_kv(
    concat_past_key_values,
    cos,
    sin,
):
    new_past_key_values = []

    for layer_idx, (key_states, value_states) in enumerate(concat_past_key_values):

        k_in = key_states

        k_rot = apply_rotary_pos_emb(k_in, cos, sin)

        key_states_rot = k_rot

        new_past_key_values.append((key_states_rot, value_states))

    return tuple(new_past_key_values)

def main():

    global_tokenizer = AutoTokenizer.from_pretrained("training_res/sum/sum_0_prompt/checkpoint-6000")

    global_model = AutoModelForCausalLM.from_pretrained("training_res/sum/sum_0_prompt/checkpoint-6000", torch_dtype=torch.bfloat16)
    global_model.to('cuda')

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path="training_res/sum/sum_0_prompt/checkpoint-6000")

    emb = LlamaRotaryEmbedding(
        dim=config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta
    ).to(device=global_model.device, dtype=torch.float32)

    batch_size = 10
    sequence_length = 100
    reencode_num = 1

    vocab_size = 128255

    sys_ids = np.random.randint(0, vocab_size, size=10).tolist()

    user_ids = np.random.randint(0, vocab_size, size=10).tolist()

    sum_ids = [128011] * reencode_num

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        device=global_model.device
    )

    with torch.no_grad():
        out = global_model(input_ids)
        old_kv = out.past_key_values

    start_time = time.time()

    concat_kv = concat_past_key_values(old_kv)
    cache_position_ids, sum_position_ids = create_position_ids(
        batch_size=batch_size,
        seq_length=sequence_length,
        start_pos=len(sys_ids),
        reencode_num=reencode_num
    )
    cos, sin = emb(x=concat_kv[0][0].to(dtype=torch.float32), position_ids=torch.tensor([cache_position_ids],device=global_model.device))
    new_kv = apply_rotary_to_concat_kv(
        concat_kv,
        cos=cos,
        sin=sin
    )
    generate_id = sys_ids + [128254]+ sum_ids * batch_size + [128255] + user_ids
    sum_position_ids = list(range(0, len(sys_ids) + 1)) + sum_position_ids + list(range(sum_position_ids[-1] + 1, sum_position_ids[-1] + 1 + len(user_ids) + 1))

    input_matrix = torch.triu(torch.full((len(generate_id), len(generate_id)), float('-inf'), dtype=torch.bfloat16, device = global_model.device), diagonal= 1)
    cache_matrix = torch.full((len(generate_id), batch_size * sequence_length), float("-inf"), dtype=torch.bfloat16, device = global_model.device)
    for idx in range(batch_size):
        cache_matrix[len(sys_ids) + 1 : len(sys_ids) + 1 + reencode_num, :idx * sequence_length] = float(0)

    attention_matrix = torch.cat([cache_matrix, input_matrix], dim = 1).unsqueeze(0).unsqueeze(0)
    print(len(generate_id))
    print(new_kv[0][0].shape)
    print(len(sum_position_ids))
    print(attention_matrix.shape)

    with torch.no_grad():
        global_model.generate(
            input_ids=torch.tensor([generate_id], device=global_model.device),
            past_key_values=new_kv,
            attention_mask=attention_matrix,
            position_ids = torch.tensor([sum_position_ids], device=global_model.device),
            cache_position = torch.tensor([cache_position_ids], device=global_model.device),
            use_cache=True,
            max_new_tokens=1,
            do_sample=False,
        )
    end_time = time.time()
    print("time:", end_time - start_time)
if __name__ == "__main__":
    main()
