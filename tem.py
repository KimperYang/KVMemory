import torch
import pandas as pd
import string
import regex
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def merge_dynamic_cache(cache_list):
    layer_num = len(cache_list[0])
    cache = cache_list[0]

    for l_idx in range(layer_num):
            cache.key_cache[l_idx] = torch.cat(
                tensors=[cache.key_cache[l_idx]] + [cache_list[b_idx].key_cache[l_idx] for b_idx in range(1, len(cache_list))],
                dim=2
            )
            cache.value_cache[l_idx] = torch.cat(
                tensors=[cache.value_cache[l_idx]] + [cache_list[b_idx].value_cache[l_idx] for b_idx in range(1, len(cache_list))],
                dim=2
            )

    return cache

def rotate_half(x):
    """
    transformers.models.llama.modeling_llama.rotate_half
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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

def apply_rerotary_position_embeddings(pkv, emb):

    device = pkv.key_cache[0].device
    emb.to(device=device)
    position_ids = torch.arange(start=0, end=pkv.key_cache[0].size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.key_cache[0].size(0), 1])
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    print(position_ids)

    for i in range(0, len(pkv.key_cache)):
        pkv.key_cache[i] = apply_rotary_pos_emb(
            k=pkv.key_cache[i].to(dtype=torch.float32), cos=cos, sin=-sin, position_ids=position_ids
        )

    return pkv

def apply_pkv_rotary_position_embeddings(pkv, start_position, emb):

    device = pkv.key_cache[0].device
    # position_ids = torch.arange(start=start_position, end=start_position + pkv.key_cache[0].size(-2), dtype=torch.int64, device=device)
    position_ids = torch.arange(start=0, end=pkv.key_cache[0].size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.key_cache[0].size(0), 1])
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)

    for i in range(0, len(pkv.key_cache)):
        pkv.key_cache[i] = apply_rotary_pos_emb(
            k=pkv.key_cache[i], cos=cos, sin=sin, position_ids=position_ids
        )
    return pkv

def generate_unrotated_memory(memory_list, model, tokenizer, emb):

    kv_list = []
    id_list = []

    for st in memory_list:
        tem_id = tokenizer(st, add_special_tokens = False, return_tensors = "pt").input_ids.to(model.device)
        id_list.append(tem_id)
        output = model(
            input_ids=tem_id.to(model.device), use_cache=True, past_key_values=DynamicCache()
        )
        raw_tem_kv = apply_rerotary_position_embeddings(pkv=output.past_key_values, emb=emb)
        kv_list.append(raw_tem_kv)
    
    
    return kv_list, id_list

def rotate_and_merge(kv_list, start_position, emb):

    cache = merge_dynamic_cache(kv_list)

    cache = apply_pkv_rotary_position_embeddings(pkv=cache, start_position=start_position, emb=emb)
    return cache


def main():
    pos = 0
    ckpt = 30000
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_id = f"/mnt/data/jingbo/kv_dump_bias_30000steps_bsz64_5e-6_full/checkpoint-{ckpt}"

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    emb = LlamaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        ).to(device=model.device, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
    total_num = 1
    correct_num = 0
    res_list = []
    
    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>"

    for i in range(total_num):

        memory_list = [sys] 

        for j in range(0,10):
            title = jsonObj["ctxs"][i][j]["title"]
            text = jsonObj["ctxs"][i][j]["text"]
            memory_list.append("<MEM_START>" + f"Document [{j+1}](Title: {title}) {text}" + "\n<MEM_END>")
        
        # Return Dynamic Cache List and an Id List
        raw_kv_list, mem_id_list = generate_unrotated_memory(memory_list, model, tokenizer, emb)

        # sys_id = tokenizer(sys, add_special_tokens= False, return_tensors="pt").input_ids.to(model.device)
        # start_position = sys_id.size(1)

        # Set timer
        rotated_mem_kv = rotate_and_merge(raw_kv_list, 0, emb)

        # sys_cache = model(input_ids = sys_id, use_cache=True, past_key_values=DynamicCache()).past_key_values
        # concat_cache = merge_dynamic_cache([sys_cache, rotated_mem_kv])
        concat_cache = rotated_mem_kv

        for l_idx in range(len(concat_cache.key_cache)):
            concat_cache.key_cache[l_idx] = concat_cache.key_cache[l_idx].to(dtype=torch.bfloat16)

        new_prompt = "<MEM_SUM><|start_header_id|>user<|end_header_id|>\n\nWrite a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Question: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        concat_id = torch.cat(mem_id_list + [prompt_id], dim=1).to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                input_ids=concat_id,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=concat_cache,
                use_cache=True
            )
        generated_seq = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        response = generated_seq[0].split('assistant\n\n')[-1]
        print(response)

        score = best_subspan_em(response, jsonObj["answers"][i])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": jsonObj["question"][i], "response": response, "gold_answer": jsonObj["answers"][i], "Score": score})
        print("Correct progress", correct_num)

if __name__ == "__main__":
    main()

# model(input_ids=block_input_ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True)