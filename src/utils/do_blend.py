import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def do_blend(model, old_kv, golden_kv, recompute_ratio, first_layer_states, position_ids, config):

    blended_kv = []
    blended_kv.append(golden_kv[0])
    blended_kv.append(golden_kv[1])
    # First pick the important indices based on the value difference in the first layer
    sequence_length = old_kv[0][0].size(2)
    topk_num = int(sequence_length * recompute_ratio)

    first_layer_old_value = old_kv[1][1]
    first_layer_golden_value = golden_kv[1][1]

    temp_diff = torch.sum((first_layer_golden_value - first_layer_old_value)**2, dim=[0,1,3]) #remain the sequence length dimension
    top_indices = torch.topk(temp_diff, k=topk_num).indices
    # print(top_indices)
    top_indices, _ = torch.sort(top_indices)
    # print(top_indices)

    # if "head_dim" in config.keys():
    #     head_dim = config.head_dim
    # else:
    head_dim = config.hidden_size // config.num_attention_heads
    # Second, starting from the second layer, we need to use the query states on selected indices and the key states in the old_kv to do the attention
    for layer_idx in range(2, len(old_kv)):
        if layer_idx == 2:
            hidden_states = first_layer_states
            hidden_states = hidden_states[:, top_indices, :]
            position_ids = position_ids[:, top_indices]


        residual = hidden_states

        transformer_block = model.model.layers[layer_idx]

        hidden_states = transformer_block.input_layernorm(hidden_states)

        bsz, q_len, _ = hidden_states.size()

        query_states = transformer_block.self_attn.q_proj(hidden_states)
        key_states = transformer_block.self_attn.k_proj(hidden_states)
        value_states = transformer_block.self_attn.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, head_dim).transpose(1, 2)

        cos, sin = transformer_block.self_attn.rotary_emb(value_states, position_ids)
        # print(position_ids)
        # print(transformer_block.self_attn.rotary_emb.rope_type)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        updated_old_k = old_kv[layer_idx][0].clone()
        updated_old_v = old_kv[layer_idx][1].clone()

        updated_old_k[:, :, top_indices, :] = key_states
        updated_old_v[:, :, top_indices, :] = value_states

        blended_kv.append((updated_old_k, updated_old_v))

        key_states = repeat_kv(updated_old_k, transformer_block.self_attn.num_key_value_groups)
        value_states = repeat_kv(updated_old_v, transformer_block.self_attn.num_key_value_groups)
        # print(key_states.shape, value_states.shape, query_states.shape)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = transformer_block.self_attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states = transformer_block.post_attention_layernorm(hidden_states)
        hidden_states = transformer_block.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return tuple(blended_kv)

def append_kv(kv_list):

    if not kv_list:
        raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")

    num_layers = len(kv_list[0])

    concatenated_past_key_values = ()

    for layer in range(num_layers):

        keys_list = [kv[layer][0] for kv in kv_list]
        values_list = [kv[layer][1] for kv in kv_list]

        concatenated_keys = torch.cat(keys_list, dim=2)
        concatenated_values = torch.cat(values_list, dim=2)

        concatenated_past_key_values = concatenated_past_key_values + ((concatenated_keys, concatenated_values),)

    return concatenated_past_key_values


# A unit test to check whether the blended kv is equal to the golden kv when recompute ratio is 100%

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# texts = ["Hello world! This is an example.", "Another sentence follows."]

# start_pos = 0
# concat_ids = []
# kv_list = []
# current_pos = start_pos

# for st in texts:
#     input_ids = tokenizer(st, add_special_tokens=False).input_ids
#     position_ids = list(range(current_pos, current_pos + len(input_ids)))
#     with torch.no_grad():
#         output = model(input_ids=torch.tensor([input_ids],device=model.device), 
#                        position_ids=torch.tensor([position_ids],device=model.device))
#     kv_list.append(output.past_key_values)

#     concat_ids += input_ids
#     current_pos += len(input_ids)

# old_kv = append_kv(kv_list)

# global_position_ids = torch.tensor([list(range(start_pos, start_pos + len(concat_ids)))],device=model.device)
# with torch.no_grad():
#     output = model(input_ids=torch.tensor([concat_ids],device=model.device),
#                     position_ids=global_position_ids,
#                     output_hidden_states=True)

# golden_kv = output.past_key_values
# first_layer_states = output.hidden_states[2]

# blend_kv = do_blend(model=model, old_kv=old_kv, golden_kv=golden_kv, recompute_ratio=0.18,first_layer_states=first_layer_states, position_ids=global_position_ids, config=config)

# for layer in range(len(blend_kv)):
#     diff = 0
#     for idx in range(blend_kv[0][0].size(2)):
#         diff += torch.sum((blend_kv[layer][0][:, :, idx, :] - golden_kv[layer][0][:, :, idx, :])**2)
#     print("layer", layer, "diff", diff.item())
