import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from datasets import load_dataset
import pandas as pd    
import json
import datetime
import string
from typing import List
from tqdm import tqdm
import regex
from llmlingua import PromptCompressor
import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')

parser.add_argument('--reencode', type=int, required=True, help='Reencode num')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt

reencode_num = args.reencode

data_list=load_dataset("hotpotqa/hotpot_qa", 'distractor', split='validation')

global_tokenizer = AutoTokenizer.from_pretrained(f"{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

# llm_lingua = PromptCompressor(
#     model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
#     use_llmlingua2=True, # Whether to use llmlingua-2
# )
llm_lingua = PromptCompressor()

def append_kv(kv_list, d):  #d=0 batch size; d=2 sequence length
    num_layers = len(kv_list[0])
    concatenated_past_key_values = ()

    for layer in range(num_layers):
        keys_list = [kv[layer][0].detach() for kv in kv_list]
        values_list = [kv[layer][1].detach() for kv in kv_list]

        concatenated_keys = torch.cat(keys_list, dim=d)
        concatenated_values = torch.cat(values_list, dim=d)
        concatenated_past_key_values += ((concatenated_keys, concatenated_values),)

    return concatenated_past_key_values

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

def construct_inference_inputs(system_ids, doc_id_list, user_ids, special_token_start, reencode_num):
    doc_ids = []
    for i in range(len(doc_id_list)):
        doc_ids.extend(doc_id_list[i])
    link_ids = [special_token_start + i for i in range(len(doc_id_list) * reencode_num)]
    input_ids = doc_ids + system_ids + link_ids +  user_ids

    new_prompt_ids = system_ids + link_ids + user_ids
    attention_matrix_right = torch.triu(torch.full((len(new_prompt_ids), len(new_prompt_ids)), float('-inf'), dtype=torch.bfloat16), diagonal= 1)
    attention_matrix_left = torch.full((len(new_prompt_ids), len(input_ids) - len(new_prompt_ids)), float('-inf'), dtype=torch.bfloat16)

    cache_position_ids = []
    input_position_ids = list(range(len(system_ids)))
    current_position = len(system_ids)

    current_row_idx = len(system_ids)
    current_col_idx = 0
    for i in range(len(doc_id_list)):
        attention_matrix_left[current_row_idx : current_row_idx + reencode_num , : current_col_idx + len(doc_id_list[i])] = float(0)
        current_row_idx = current_row_idx + reencode_num
        current_col_idx = current_col_idx + len(doc_id_list[i])

        cache_position_ids.extend(list(range(current_position, current_position + len(doc_id_list[i]))))
        input_position_ids.extend(list(range(current_position + len(doc_id_list[i]), current_position + len(doc_id_list[i]) + reencode_num)))
        current_position += len(doc_id_list[i]) + reencode_num

    input_position_ids.extend(list(range(current_position, current_position + len(user_ids))))

    attention_matrix_left[current_row_idx : ,] = float(0)
    
    attention_matrix = torch.cat([attention_matrix_left, attention_matrix_right], dim = 1)

    return input_ids, attention_matrix, cache_position_ids, input_position_ids, new_prompt_ids,

def main():
    global_model.to('cuda')

    special_token_start=128011
    mem_start=128254
    mem_end=128255

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    total_num = len(data_list)
    correct_num = 0
    res_list = []

    for i in tqdm(range(total_num)):
        memory_list = []

        for j in range(len(data_list[i]['context']['title'])):
            title = data_list[i]['context']['title'][j]
            text = ''.join(data_list[i]['context']['sentences'][j])
            compressed_text = llm_lingua.compress_prompt(text, rate=0.5, force_tokens = ['\n', '?'])['compressed_prompt']
            memory_list.append(f"Document [{j+1}](Title: {title}) {compressed_text}\n")

        # memory_list.insert(0, template)
        sys_ids = global_tokenizer(template, add_special_tokens=False).input_ids
        sys_ids = sys_ids + [mem_start]

        current_position = len(sys_ids)
        kv_list = []
        doc_id_list = []
        for j in range(len(memory_list)):

            tem_ids = global_tokenizer(memory_list[j], add_special_tokens=False).input_ids
            doc_id_list.append(tem_ids)
            tem_position_ids = list(range(current_position, current_position + len(tem_ids)))
            current_position += len(tem_ids) + reencode_num
            tem_kv = global_model(input_ids = torch.tensor([tem_ids], device=global_model.device),
                                  position_ids = torch.tensor([tem_position_ids], device=global_model.device)
                                  ).past_key_values
            kv_list.append(tem_kv)


        user_prompt = data_list[i]['question'] + "<|eot_id|>"
        user_ids = [mem_end] + global_tokenizer(user_prompt, add_special_tokens=False).input_ids

        input_ids, attention_matrix, cache_position_ids, input_position_ids, new_prompt_ids = construct_inference_inputs(sys_ids, doc_id_list, user_ids, special_token_start, reencode_num)

        input_ids = torch.tensor([input_ids], device=global_model.device)
        attention_matrix = attention_matrix.unsqueeze(0).unsqueeze(0).to(device=global_model.device)
        cache_position_ids = torch.tensor([cache_position_ids], device=global_model.device)
        input_position_ids = torch.tensor([input_position_ids], device=global_model.device)
        new_prompt_ids = torch.tensor([new_prompt_ids], device=global_model.device)
        concat_kv = append_kv(kv_list, 2)

        asst_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        asst_ids = global_tokenizer(asst_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(global_model.device)
        generate_id = torch.cat([input_ids, asst_ids], dim = 1)

        global_model.eval()
        with torch.no_grad():
            outputs = global_model(input_ids = new_prompt_ids, 
                                   past_key_values = concat_kv,
                                   attention_mask = attention_matrix,
                                   cache_position = cache_position_ids,
                                   position_ids = input_position_ids,
                                   use_cache=True,
                                   )
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
        print(data_list[i]['question'])
        print(response)

        score = best_subspan_em(response, [data_list[i]['answer']])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data_list[i]['question'], "response": response, "gold_answer": [data_list[i]['answer']], "Score": score})
        print("Accuracy", correct_num / (i+1))

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/lingua1/HQA_ckpt{ckpt}_{accuracy}_{time_str}_{reencode_num}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
