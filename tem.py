import json

def compare_jsonl_responses(file1, file2):
    """
    Compare the 'response' values in two JSONL files.

    Args:
        file1 (str): Path to the first JSONL file.
        file2 (str): Path to the second JSONL file.

    Returns:
        bool: True if all 'response' values match, False otherwise.
        list: List of indices where the 'response' values differ.
    """
    differing_indices = []

    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = [json.loads(line)['response'] for line in f1 if 'response' in json.loads(line)]
            lines2 = [json.loads(line)['response'] for line in f2 if 'response' in json.loads(line)]
        
        # Ensure both files have the same number of items
        if len(lines1) != len(lines2):
            print("The files have different numbers of items.")
            return False, None

        # Compare 'response' values
        for i, (resp1, resp2) in enumerate(zip(lines1, lines2)):
            if resp1 != resp2:
                differing_indices.append(i)
        
        return len(differing_indices) == 0, differing_indices

    except Exception as e:
        print(f"Error: {e}")
        return False, None

# Example usage
file1 = "/home/jingbo/KVMemory/result/11-18/nq/nq_llama3.2_1B_bias_bsz64_50000steps_2e-5_full_at0_0.352_20241119-191302.jsonl"
file2 = "/home/jingbo/KVMemory/result/11-18/nq/nq_llama3.2_1B_mix5_bsz64_50000steps_2e-5_full_at0_0.352_20241119-011651.jsonl"

are_same, diff_indices = compare_jsonl_responses(file1, file2)
if are_same:
    print("All 'response' values are the same.")
else:
    print(f"'response' values differ at indices: {diff_indices}")
