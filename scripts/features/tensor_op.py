import torch
# from src.utils.cache import concat_kv
# Example input tensor

mum_len = 20
bsz = 2
each_mem_len = 4
memory_position = torch.arange(1, 1 + mum_len).unsqueeze(0)
memory_positions = torch.cat([memory_position] * bsz)
memory_position_batch = memory_positions.reshape(-1, each_mem_len)

print(memory_position_batch)
batch_input_ids = torch.tensor([[1, 2],[3, 4],
                                [5, 6],[7, 8]])

# # Apply torch.split with split_size of 1 along dim=1 (sequence length dimension)
# splits = torch.split(batch_input_ids, 2, dim=1)

# # Print each split tensor
# for split in splits:
#     print(split)

# Reshape the tensor
reshaped_tensor = batch_input_ids.view(2,4)

# Print the result
# print(reshaped_tensor)

# def concat_kv(split_kv, num_memory):
#     num_layers = len(split_kv)
#     split_batch_size = split_kv[0][0].size(0)
#     final_past_key_values = ()

#     for layer_idx in range(num_layers):
#         key_cache, value_cache = split_kv[layer_idx]

#         concatenated_keys_list = []
#         concatenated_values_list = []

#         for i in range(0, split_batch_size, num_memory):
            
#             key_group = key_cache[i:i+num_memory]
#             key_list = torch.split(key_group, 1, dim=0)
#             value_group = value_cache[i:i+num_memory]
#             value_list = torch.split(value_group, 1, dim=0)

#             concatenated_key = torch.cat(key_list, dim=2)
#             concatenated_value = torch.cat(value_list, dim=2)

#             concatenated_keys_list.append(concatenated_key)
#             concatenated_values_list.append(concatenated_value)

#         layer_concatenated_keys = torch.cat(concatenated_keys_list, dim=0)  # Concatenate along batch dimension
#         layer_concatenated_values = torch.cat(concatenated_values_list, dim=0)

#         final_past_key_values += ((layer_concatenated_keys, layer_concatenated_values),)
        
#     return final_past_key_values

# rand_tensor = torch.randn(100, 8, 20, 16) #it should be converted to (3, 8, 60, 16)
# kv_cache = ((rand_tensor,rand_tensor),
#             (rand_tensor,rand_tensor),
#             (rand_tensor,rand_tensor),
#             (rand_tensor,rand_tensor),
#             (rand_tensor,rand_tensor),
#             (rand_tensor,rand_tensor),
#             )

# res = concat_kv(kv_cache, 20)
# print(type(res), len(res))
# print(type(res[0]), len(res[0]))
# print(type(res[0][0]), res[0][0].shape)
# print(kv_cache[0][0].shape)