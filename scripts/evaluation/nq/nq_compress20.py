import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
import string
from typing import List
from src.data.compress import insert_mem_tokens, get_position_id, construct_compress_attention_matrix
import regex

import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
parser.add_argument('--pos', type=int, required=True, help='Position value')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt
pos = args.pos

if pos in [0, 4, 9]:
    jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
else:
    jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)

global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

def filter_id(input_ids, intervals_to_remove):  

    T = input_ids.shape[1] 
    mask = torch.ones(T, dtype=bool)

    for (start, end) in intervals_to_remove:
        mask[start:end] = False  # set these indices to False

    return input_ids[:, mask]

def filter_kv(past_key_values, intervals_to_remove):
    num_layers = len(past_key_values)
    filtered_past_key_values = ()    

    T = past_key_values[0][0].shape[2] 
    mask = torch.ones(T, dtype=bool)

    for (start, end) in intervals_to_remove:
        mask[start:end] = False  # set these indices to False

    for layer_id in range(num_layers):
        tem_key = past_key_values[layer_id][0]
        tem_value = past_key_values[layer_id][1]

        filtered_key = tem_key[:, :, mask, :]
        filtered_value = tem_value[:, :, mask, :]

        filtered_past_key_values += ((filtered_key, filtered_value),)

    return filtered_past_key_values

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

    mem_start = 128254
    mem_end = 128255
    compress_tokens = list(range(128011, 128031))

    global_model.to('cuda')

    # total_num = len(jsonObj)
    total_num = 500
    correct_num = 0
    res_list = []

    for i in range(total_num):

        template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant."
        # template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        sys_id = global_tokenizer(template, add_special_tokens=False).input_ids

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

        raw_input_ids = sys_id
        position = len(sys_id)
        biased_ranges = []

        for j in range(0,10):
            title = doc_list[j]["title"]
            text = doc_list[j]["text"]
            tem_id = global_tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
            raw_input_ids += tem_id
            biased_ranges.append([position, position + len(tem_id)])

            position += len(tem_id)

        new_ids, new_ranges = insert_mem_tokens(
            raw_input_ids, biased_ranges, compress_tokens, mem_start, mem_end
        )

        position_ids = get_position_id(new_ids, new_ranges)

        attention_matrix = construct_compress_attention_matrix(len(new_ids), new_ranges, len(new_ids), global_model.device, len(compress_tokens)).unsqueeze(0).unsqueeze(0)       

        new_ids = torch.tensor([new_ids], device = global_model.device)
        position_ids = torch.tensor([position_ids], device = global_model.device)

        with torch.no_grad():
            outputs = global_model(input_ids = new_ids, attention_mask = attention_matrix, position_ids = position_ids)
            past_key_values = outputs.past_key_values

        filtered_kv = filter_kv(past_key_values, new_ranges)
        filtered_id = filter_id(new_ids, new_ranges)

        # import ipdb
        # ipdb.set_trace()

        new_prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Question: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        # new_prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>" + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        generate_id = torch.cat([filtered_id, prompt_id], dim = 1)

        with torch.no_grad():

            outputs = global_model.generate(
                input_ids=generate_id,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=filtered_kv,
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

    file_name = f"result/{run_name}/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
