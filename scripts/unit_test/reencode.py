import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.cache import append_kv
from src.data.attention import construct_biased_attention_matrix

# Example comparing hidden states
def compare_hidden_states(hidden_states1, hidden_states2, remaining_ids_len, rtol=1e-5, atol=1e-5):
    """
    Compare two sets of hidden states with specified tolerances
    
    Args:
        hidden_states1: First set of hidden states (tuple/list of tensors)
        hidden_states2: Second set of hidden states (tuple/list of tensors)
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    # Verify same number of layers
    assert len(hidden_states1) == len(hidden_states2), "Hidden states have different lengths"
    
    # Compare each layer's hidden states
    for layer_idx, (h1, h2) in enumerate(zip(hidden_states1, hidden_states2)):
        h1 = h1[:, -remaining_ids_len:, :]
        h2 = h2[:, -remaining_ids_len:, :]
        try:
            torch.testing.assert_close(
                h1,
                h2,
                rtol=rtol,  # Relative tolerance
                atol=atol,  # Absolute tolerance
                check_device=True,  # Check if tensors are on same device
                check_dtype=True    # Check if tensors have same dtype
            )
            print(f"Layer {layer_idx}: Hidden states match within tolerances")
        except AssertionError as e:
            print(f"Layer {layer_idx}: Hidden states differ!")
            print(e)
            
            # Optional: Print more detailed statistics
            max_diff = torch.max(torch.abs(h1 - h2))
            mean_diff = torch.mean(torch.abs(h1 - h2))
            print(f"Max difference: {max_diff}")
            print(f"Mean difference: {mean_diff}")


run_name = "kv_dump_bias_50000steps_bsz64_2e-5_full"
ckpt = 50000

model_name = f"/mnt/data/jingbo/{run_name}/checkpoint-{ckpt}"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float32, attn_implementation = 'sdpa')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to('cuda')

# mem_list = ["<MEM_START>Ronald scored 2 in the last game.<MEM_END>", 
#             "<MEM_START>Messi scored 3 in the last game.<MEM_END>"]

mem_list = [
    "Hello World",
    "Hello Hi World",
    "Hello World",
]

mem_num = len(mem_list)

mem_len = []



for mem in mem_list:
    tem_id = tokenizer(mem, add_special_tokens=False, return_tensors="pt").input_ids
    mem_len.append(tem_id.size(1))


question = "<MEM_SUM>" * mem_num + "Hello World"
q_id = tokenizer(question, add_special_tokens= False, return_tensors="pt").input_ids

sys_id = tokenizer("<|begin_of_text|>", add_special_tokens=False, return_tensors="pt").input_ids
sys_len = sys_id.size(1)

prompt_id = torch.cat([sys_id, q_id], dim = 1)

prompt_length = prompt_id.size(1)
q_attention_matrix = torch.triu(torch.full((prompt_length, prompt_length), float('-inf'), dtype=torch.float32), diagonal= 1)

kv_attention_matrix = torch.full((prompt_length, sum(mem_len)), float(0), dtype=torch.float32)

kv_attention_matrix[0:sys_len, :] = float('-inf')

for i in range(mem_num):
        
    kv_attention_matrix[sys_len + i][sum(mem_len[:i + 1]):] = float('-inf')

final_attention_matrix = torch.cat([kv_attention_matrix, q_attention_matrix], dim = 1)

#  Calculate KV

kv_list = []
for i in range(len(mem_list)):
    tem_id = tokenizer(mem_list[i], add_special_tokens= False, return_tensors="pt").input_ids
    start_pos = sys_len + sum(mem_len[:i]) + i
    p_id = torch.arange(start_pos, start_pos + tem_id.size(1)).unsqueeze(0)
    output = model(input_ids = tem_id.to(model.device), position_ids = p_id.to(model.device))
    kv_list.append(output.past_key_values)

mem_kv = append_kv(kv_list, 2)

prompt_pos = torch.arange(prompt_length)

for idx in range(prompt_length):
    if idx < sys_len:
        prompt_pos[idx] = prompt_pos[idx] 
    elif sys_len <= idx < sys_len + mem_num:
        prompt_pos[idx] += sum(mem_len[: idx - sys_len + 1])
    elif idx >= sys_len + mem_num:
        prompt_pos[idx] += sum(mem_len)

prefill_output = model(input_ids = prompt_id.to(model.device), attention_mask = final_attention_matrix.unsqueeze(0).unsqueeze(0).to(model.device), past_key_values = mem_kv, position_ids = prompt_pos.unsqueeze(0).to(model.device), use_cache = True, output_hidden_states=True)

# print(prefill_kv[0][0].shape)

# implement training case
current_index = sys_len
biased_index = []
id_list = [sys_id]

for st in mem_list:
    tem_id = tokenizer(st + "<MEM_SUM>", add_special_tokens= False, return_tensors="pt").input_ids
    id_list.append(tem_id)
    biased_index.append([current_index, current_index + tem_id.size(1) - 1])
    current_index += tem_id.size(1)

id_list.append(tokenizer("Hello World", add_special_tokens= False, return_tensors="pt").input_ids)

concat_id = torch.cat(id_list, dim = 1)

attention_matrix = construct_biased_attention_matrix(concat_id.size(1), biased_index, concat_id.size(1), model.device)
print(attention_matrix)

baseline_output = model(input_ids = concat_id.to(model.device), attention_mask = attention_matrix.unsqueeze(0).unsqueeze(0), output_hidden_states=True)

hidden_states1 = prefill_output.hidden_states
hidden_states2 = baseline_output.hidden_states
loss_id_len = 2 
# print(loss_id_len)
# print(hidden_states1[-1][:, -loss_id_len:, :].shape)
# print(hidden_states2[-1].shape)

compare_hidden_states(hidden_states1, hidden_states2, loss_id_len)