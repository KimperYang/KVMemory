import torch
import copy

def insert_mem_tokens(
    original_list,    # e.g. [1,2,3,4,5,6,7,8]
    ranges,           # list of [start, end) half-open intervals
    new_token,        # e.g. [100, 1000]
    start_number,     # e.g. -111
    end_number        # e.g. -999
):
    """
    Insert:
      - `start_number` before the *first* range slice
      - `new_token`    after *each* range slice
      - `end_number`   after the *last* range slice
    Return (new_list, new_ranges) 
      where new_ranges are the half-open intervals adjusted in the new list.
    """
    result_list = []
    new_ranges = []
    offset = 0
    
    # Keep track of the last included index from old list.
    # For half-open [s,e), the last included old index is e-1.
    prev_included = -1
    
    # Sort ranges by their start just in case
    # sorted_ranges = sorted(ranges, key=lambda x: x[0])
    sorted_ranges =ranges

    for i, (start, end) in enumerate(sorted_ranges):
        is_first_range = (i == 0)
        is_last_range  = (i == len(sorted_ranges) - 1)
        
        # 1) Append all elements between the last included and (start-1)
        result_list.extend(original_list[prev_included + 1 : start])
        
        # 2) If this is the first range, insert the start_number
        if is_first_range:
            result_list.append(start_number)
            offset += 1  # We inserted 1 extra item in the new list
        
        # 3) Append the slice [start, end)
        result_list.extend(original_list[start : end])
        
        # 4) Record the new half-open range [start+offset, end+offset)
        new_start = start + offset
        new_end   = end   + offset
        new_ranges.append([new_start, new_end])
        
        # 5) Insert the token after this range
        result_list.extend(new_token)
        offset += len(new_token)
        
        # 6) If this is the last range, insert the end_number
        if is_last_range:
            result_list.append(end_number)
            offset += 1
        
        # 7) Update prev_included
        prev_included = end - 1
    
    # 8) Append any leftover elements after the last range
    result_list.extend(original_list[prev_included + 1 : ])
    
    return result_list, new_ranges

def get_position_id(id_list, ranges):

    current_position = 0
    position_ids = []
    for i, (start, end) in enumerate(ranges):

        inter_chunk_length = start - len(position_ids)

        position_ids += list(range(current_position, current_position + inter_chunk_length))
        current_position += inter_chunk_length

        chunk_length = end - start

        position_ids += list(range(current_position - chunk_length, current_position))

    position_ids += list(range(current_position, current_position + len(id_list) - len(position_ids)))

    return position_ids

def construct_compress_attention_matrix(seq_len, shift_ranges, max_len, device, num_sum_tokens=20):

    attention_matrix = torch.triu(torch.full((max_len, max_len), float('-inf'), dtype=torch.bfloat16, device = device), diagonal= 1)

    if shift_ranges is not None:

        mem_end_position = shift_ranges[-1][-1] + num_sum_tokens
        for indices in shift_ranges:
            i = indices[0]
            j = indices[1]

            attention_matrix[i : j + num_sum_tokens, 0 : i] = float('-inf')
            attention_matrix[j + num_sum_tokens: , i : j] = float('-inf')

    attention_matrix[seq_len :, :] = float('-inf')
    attention_matrix[: ,seq_len :] = float('-inf')

    return attention_matrix

def construct_compress_input(input_ids, biased_index, max_len):
    mem_start = 128254
    mem_end = 128255
    compress_tokens = list(range(128011, 128013))

    new_ids, new_ranges = insert_mem_tokens(
        input_ids, biased_index, compress_tokens, mem_start, mem_end
    )

    position_id = get_position_id(new_ids, new_ranges)

    attention_matrix = construct_compress_attention_matrix(len(new_ids), new_ranges, max_len, 'cpu', len(compress_tokens))

    return new_ids, position_id, attention_matrix

lst     = [0, 1, 2, 3, 4, 5, 6, 7]
ranges  = [[2,3],[4,5]]
max_len = 15

new_ids, position_id, attention_matrix = construct_compress_input(lst, ranges, max_len)
print("new_ids: ", new_ids)
print("position_id: ", position_id)
print(attention_matrix)

# def construct_compress_input(input_ids, biased_index, max_len):
#     mem_start = 128254
#     mem_end = 128255
#     compress_tokens = list(range(128011, 128013))

#     new_ids, new_ranges = insert_mem_tokens(
#         input_ids, biased_index, compress_tokens, mem_start, mem_end
#     )

#     position_id = get_position_id(new_ids, new_ranges)

#     attention_ranges = copy.deepcopy(new_ranges)
#     for i in range(len(attention_ranges)):
#         attention_ranges[i][1] += 2

#     attention_matrix = construct_biased_attention_matrix(len(new_ids), attention_ranges, max_len, 'cpu')

#     biased_indices = []
#     current_position = 0
#     for i, (start, end) in enumerate(attention_ranges):
#         biased_indices += list(range(current_position, start))
#         current_position = end
#     biased_indices += list(range(current_position, len(new_ids)))

#     for idx in biased_indices:
#         for i, (start, end) in enumerate(new_ranges):
#             attention_matrix[idx][start:end] = float("-inf")

#     return new_ids, position_id, attention_matrix