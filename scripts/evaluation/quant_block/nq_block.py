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
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
parser.add_argument('--pos', type=int, required=True, help='Position value')
parser.add_argument('--run', type=str, required=True, help='Run name')
args = parser.parse_args()

ckpt = args.ckpt
pos = args.pos
run_name = args.run

if pos in [0, 4, 9]:
    jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
else:
    jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)

# global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)

# global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

# global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

global_tokenizer = AutoTokenizer.from_pretrained(f"{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"{run_name}/checkpoint-{ckpt}",  device_map="cuda:0", load_in_8bit=True)

# global_tokenizer = AutoTokenizer.from_pretrained(run_name)

# global_model = AutoModelForCausalLM.from_pretrained(run_name, torch_dtype=torch.bfloat16)

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

    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"

    # template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
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
            # memory_list.append(f"- Title: {title}\n{text}\n") #same as training setting in blockqa data

        memory_list.insert(0, template)

        biased_index = []
        id_list = []

        idx = 0

        for st in memory_list:

            tem_id = global_tokenizer(st, return_tensors="pt", add_special_tokens=False).input_ids
            biased_index.append([idx, idx + tem_id.size(1)])

            id_list.append(tem_id)

            idx = idx + tem_id.size(1)

        # new_prompt = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Question: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        new_prompt = jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        # id_list.append(prompt_id)

        concat_id = torch.cat(id_list, dim=1).to(global_model.device)
        attention_matrix = construct_biased_attention_matrix(concat_id.size(1), biased_index, concat_id.size(1), global_model.device).unsqueeze(0).unsqueeze(0)
        attention_matrix = attention_matrix.to(global_model.dtype) 


        global_model.eval()

        generate_id = torch.cat([concat_id, prompt_id], dim = 1)

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

    file_name = f"result/quant/block/NQ2_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"
    # file_name = f"result/{run_name}/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"
    # file_name = f"result/new_data/block_31_8B/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
