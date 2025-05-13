import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
import string
from typing import List
# from src.data.attention import construct_biased_attention_matrix
import regex

import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
parser.add_argument('--pos', type=int, required=True, help='Position value')
parser.add_argument('--reencode', type=int, required=True, help='Reencode num')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt
pos = args.pos
reencode_num = args.reencode

if pos in [0, 4, 9]:
    jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
else:
    jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)

# global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

# global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16, device_map="cuda:0", load_in_8bit=True)

global_tokenizer = AutoTokenizer.from_pretrained(f"{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16, device_map="cuda:0", load_in_8bit=True)
# vocab_size = len(global_tokenizer)
# base_model.resize_token_embeddings(vocab_size)

# peft_config_path = "/mnt/data/jingbo/kv_dump_combine_mix5_5000steps_5e-6_full/checkpoint-5000"  # Path to the directory where LoRA weights are stored

# global_model = PeftModel.from_pretrained(base_model, peft_config_path)

def construct_biased_attention_matrix(seq_len, biased_ranges, max_len, device):
    """
    Constructs a padded biased attention matrix.

    Parameters:
    - seq_len: The actual sequence length of the input.
    - biased_ranges: List of [start, end] indices defining biased position ranges.
    - max_len: The maximum sequence length for padding.

    Returns:
    - A numpy array representing the padded biased attention matrix.
    """
    # Initialize the attention matrix with -inf for masking
    attention_matrix = torch.triu(torch.full((max_len, max_len), float('-inf'), dtype=torch.bfloat16, device = device), diagonal= 1)

    if biased_ranges is not None:
        for indices in biased_ranges:
            i = indices[0]
            j = indices[1]

            attention_matrix[i : j, 0 : i] = float('-inf')

    attention_matrix[seq_len :, :] = float('-inf')
    attention_matrix[: ,seq_len :] = float('-inf')

    if  attention_matrix.max() != 0:
        print("wrong", seq_len, biased_ranges, max_len)
        print(attention_matrix)

    return attention_matrix

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

def main():

    special_token_start=128011
    mem_start=128254
    mem_end=128255

    # template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    # template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nWrite a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    # total_num = len(jsonObj)
    total_num = 500
    correct_num = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))
        memory_list = []
        doc_list = []

        for k in range(0,10):
            title = jsonObj["ctxs"][i][k]["title"]
            text = jsonObj["ctxs"][i][k]["text"]
            doc_list.append({'title': title, 'text':text})

        if pos not in [0,4,9]:
            ground_truth = doc_list.pop(0)
            doc_list.insert(pos, ground_truth)

        for j in range(0,10):
            title = doc_list[j]["title"]
            text = doc_list[j]["text"]
            memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

        # memory_list.insert(0, template)
        sys_id = global_tokenizer(template, add_special_tokens=False).input_ids
        sys_id = sys_id + [mem_start]

        biased_index = []
        concat_id = []

        idx = len(sys_id)

        for j in range(len(memory_list)):

            tem_id = global_tokenizer(memory_list[j], add_special_tokens=False).input_ids

            for sub_idx in range(reencode_num):
                tem_id = tem_id + [special_token_start + reencode_num * j + sub_idx]

            biased_index.append([idx, idx + len(tem_id) - reencode_num])

            concat_id += tem_id

            idx = idx + len(tem_id)

        concat_id = sys_id + concat_id + [mem_end]
        concat_id = torch.tensor([concat_id], device=global_model.device)
        attention_matrix = construct_biased_attention_matrix(concat_id.size(1), biased_index, concat_id.size(1), global_model.device).unsqueeze(0).unsqueeze(0)

        # new_prompt = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Question: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        new_prompt = jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        # new_prompt = "\nQuestion: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        generate_id = torch.cat([concat_id, prompt_id], dim = 1)

        global_model.eval()
        with torch.no_grad():
            outputs = global_model(input_ids = concat_id, attention_mask = attention_matrix)
            past_key_values = outputs.past_key_values

            outputs = global_model.generate(
                input_ids=generate_id,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=past_key_values,
                use_cache=True
            )
        # print(outputs)
        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]
        print(response)

        score = best_subspan_em(response, jsonObj["answers"][i])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": jsonObj["question"][i], "response": response, "gold_answer": jsonObj["answers"][i], "Score": score})
        print("Correct progress", correct_num)

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    # file_name = f"result/{run_name}/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"
    file_name = f"result/quant/sum_{reencode_num}/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
