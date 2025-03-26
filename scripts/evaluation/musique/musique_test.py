import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd    
import json
import datetime
import string
from typing import List
from src.data.attention import construct_biased_attention_matrix
import regex
import argparse
from datasets import load_dataset

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

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
    parser.add_argument('--run', type=str, required=True, help='Checkpoint number')
    parser.add_argument('--reencode', type=int, required=True, help='Reencode num')

    args = parser.parse_args()

    ckpt = args.ckpt
    run_name = args.run
    reencode_num = args.reencode

    special_token_start=128011
    mem_start=128254
    mem_end=128255

    data_list=load_dataset("dgslibisey/MuSiQue", split='validation')

    global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

    global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

    global_model.to('cuda')

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

    # total_num = len(jsonObj)
    total_num = 100
    correct_num = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))

        sys_id = global_tokenizer(template, add_special_tokens=False).input_ids
        sys_id = sys_id + [mem_start]
        memory_list = []


        for j in range(len(data_list[i]['paragraphs'])):
            title = data_list[i]['paragraphs'][j]['title']
            text = data_list[i]['paragraphs'][j]['paragraph_text']
            memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

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

        concat_id = sys_id + concat_id + [mem_end] + global_tokenizer(data_list[i]['question'], add_special_tokens = False)["input_ids"]
        attention_matrix = construct_biased_attention_matrix(len(concat_id), biased_index, len(concat_id), global_model.device).unsqueeze(0).unsqueeze(0)

        res_list.append({"input_ids":[concat_id], "attention_mask":attention_matrix.tolist()})

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()

    file_name = f"result/musique_single.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
