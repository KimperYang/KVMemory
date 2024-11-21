import torch
from datasets import load_from_disk
from src.data.mapfunc import bias_attention_preprocessor, multi_kv_preprocessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.attention import construct_biased_attention_matrix
from transformers import LlamaModel
from src.utils.cache import generate_kv_with_id, concat_kv, append_kv, generate_kv_with_position

model_name = '/mnt/data/jingbo/kv_dump_bias_30000steps_bsz64_5e-6_full/checkpoint-30000/checkpoint-30000/checkpoint-30000'
# model_name = 'meta-llama/Llama-3.2-1B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation='sdpa')
model.to('cuda')

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

preprocessor = bias_attention_preprocessor(
    tokenizer=tokenizer,
    max_len=4096
)

sftmem_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sftmem_new").select(range(100))

# new_token = ["<MEM_START>","<MEM_END>", "<MEM_SUM>"]
# tokenizer.add_tokens(new_token)
# model.resize_token_embeddings(len(tokenizer))

sftmem_block = sftmem_raw.map(
    preprocessor.process_sftmem,
    num_proc=256,
    remove_columns=["system", "mask", "dataset", "conversations"],
    batched=False,
    # load_from_cache_file=True
)

sample = sftmem_block[67]

input_ids = torch.LongTensor(sample['input_ids']).to(model.device)
attention_matrices = construct_biased_attention_matrix(len(sample['input_ids'][0]), sample['biased_index'], len(sample['input_ids'][0]), model.device).unsqueeze(0).unsqueeze(0)
labels = torch.LongTensor(sample['labels']).to(model.device)
# print(input_ids.shape, attention_matrices.shape, labels.shape)
# with torch.no_grad():
output1 = model(input_ids = input_ids, attention_mask = attention_matrices, labels = labels, use_cache = False, output_hidden_states=True)
print(output1.loss)
hidden_states1 = output1.hidden_states
last_hidden_state1 = hidden_states1[-1]
# print(last_hidden_state1.shape)

preprocessor2 = multi_kv_preprocessor(
    tokenizer=tokenizer,
    max_len=4096
)

sftmem_kv = sftmem_raw.map(
    preprocessor2.process_sftmem,
    num_proc=256,
    remove_columns=["system", "mask", "dataset", "conversations"],
    batched=False,
    # load_from_cache_file=True
)

sample = sftmem_kv[67]

# for key in sample.keys():
#     print(key)
#     print(sample[key])
num_memory = len(sample['memory_position'])
sys_key_values = generate_kv_with_id(model, torch.LongTensor(sample['sys_id']))
kv_list = [sys_key_values]

for idx in range(num_memory):
    kv_list.append(generate_kv_with_position(model, torch.LongTensor([sample['split_memory_id'][idx]]), position_ids = torch.LongTensor([sample['memory_position'][idx]])))
past_key_values = append_kv(kv_list, 2)

output2 = model(input_ids=torch.LongTensor(sample['input_ids']).to(model.device), labels = torch.LongTensor(sample['labels']).to(model.device), past_key_values = past_key_values, use_cache = True, output_hidden_states=True)

print(output2.loss)
hidden_states2 = output2.hidden_states
loss_id_len = len(sample['input_ids'][0])
# print(loss_id_len)
# print(hidden_states1[-1][:, -loss_id_len:, :].shape)
# print(hidden_states2[-1].shape)

compare_hidden_states(hidden_states1, hidden_states2, loss_id_len)