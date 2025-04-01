import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import time
from typing import List
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM
import numpy as np

def move_cache_device(kv, device):
    for layer in range(len(kv)):
        kv[layer][0].to(device)
        kv[layer][1].to(device)

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

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16)
    global_model.to('cuda')

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct")

    emb = LlamaRotaryEmbedding(
        dim=config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta
    ).to(device=global_model.device, dtype=torch.bfloat16)

    batch_size = 10
    sequence_length = 500

    vocab_size = 128255
    total_time = 0

    for i in range(110):
        sys_ids = np.random.randint(0, vocab_size, size=10).tolist()

        user_ids = np.random.randint(0, vocab_size, size=10).tolist()

        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, sequence_length),
            device=global_model.device
        )

        start_time = time.time()
        with torch.no_grad():
            prefill_output = global_model(
                input_ids=torch.tensor([sys_ids + list(input_ids.view(-1))+user_ids], device=global_model.device),
                use_cache=True,
            )

        end_time = time.time()
        print("time:", end_time - start_time)
        if i>= 10:
            total_time += end_time - start_time
    print('Avg time:', total_time/100)
if __name__ == "__main__":
    main()
