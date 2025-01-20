import pytest
from src.data.compress import insert_mem_tokens, get_position_id, construct_compress_attention_matrix

mem_start = 128254
mem_end = 128255
compress_tokens = list(range(128011, 128013))

input_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
biased_ranges = [[3,5], [5,7]]

def test_insert_mem_tokens(input_ids, biased_ranges):
    new_ids, new_ranges = insert_mem_tokens(input_ids, biased_ranges, compress_tokens, mem_start, mem_end)
    assert len(biased_ranges) == len(new_ranges)
    assert len(new_ids) - len(input_ids) == 2 + len(biased_ranges) * len(compress_tokens)

    for mem_idx in range(len(biased_ranges)):
        start1 = biased_ranges[mem_idx][0]
        end1 = biased_ranges[mem_idx][1]

        start2 = new_ranges[mem_idx][0]
        end2 = new_ranges[mem_idx][1]

        assert input_ids[start1:end1] == new_ids[start2:end2]

def test_get_position_id(input_ids, biased_ranges):
    new_ids, new_ranges = insert_mem_tokens(input_ids, biased_ranges, compress_tokens, mem_start, mem_end)
    position_id = get_position_id(new_ids, new_ranges)

    assert len(position_id) == len(new_ids)

    for mem_idx in range(len(new_ranges)):
        current_start_idx = new_ranges[mem_idx][0]
        current_end_idx = new_ranges[mem_idx][1] - 1

        # print(position_id)
        # print(position_id[current_start_idx: current_end_idx + 1])
        # print(list(range(position_id[current_start_idx], position_id[current_end_idx] + 1)))

        assert position_id[current_start_idx: current_end_idx + 1] == list(range(position_id[current_start_idx], position_id[current_end_idx] + 1))
        
        assert position_id[current_end_idx + 1] == position_id[current_start_idx - 1] + 1

def test_construct_compress_attention_matrix(input_ids, biased_ranges):
    new_ids, new_ranges = insert_mem_tokens(input_ids, biased_ranges, compress_tokens, mem_start, mem_end)
    attention_matrix = construct_compress_attention_matrix(len(new_ids), new_ranges, len(new_ids), 'cpu', len(compress_tokens))

    for mem_idx in new_ranges:
        for row_idx in range(attention_matrix.size(0)):
            if row_idx < mem_idx[0]:
                assert (attention_matrix[row_idx, mem_idx[0]:mem_idx[1]] == float('-inf')).all()
            
            if mem_idx[0] <= row_idx < mem_idx[1] + len(compress_tokens):
                assert (attention_matrix[row_idx, mem_idx[0]:row_idx] == float(0)).all()
                assert (attention_matrix[row_idx, row_idx + 1:mem_idx[1]] == float('-inf')).all()
            
            if row_idx >= new_ranges[-1][-1] + len(compress_tokens):
                # print(row_idx)
                # print(attention_matrix[row_idx])
                # print()
                assert (attention_matrix[row_idx, mem_idx[0]:mem_idx[1]] == float('-inf')).all() 
                assert (attention_matrix[row_idx, mem_idx[1]:mem_idx[1] + len(compress_tokens)] == float(0)).all()

test_insert_mem_tokens(input_ids, biased_ranges)
test_get_position_id(input_ids, biased_ranges)
test_construct_compress_attention_matrix(input_ids, biased_ranges)