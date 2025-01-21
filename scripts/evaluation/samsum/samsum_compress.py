import argparse
import datetime
import json

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.attention import construct_biased_attention_matrix
from src.data.compress import insert_mem_tokens, get_position_id, construct_compress_attention_matrix

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

def calculate_rouge_l_score(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    # rouge_l_score = scores['rougeL'].recall
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score

def main():

    mem_start = 128254
    mem_end = 128255
    compress_tokens = list(range(128011, 128031))

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
    parser.add_argument('--run', type=str, required=True, help='Run name')

    args = parser.parse_args()

    ckpt = args.ckpt
    run_name = args.run

    global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

    global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)
    global_model.to('cuda')

    samsum = load_dataset("Samsung/samsum")

    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSummarize the dialogue into a few short sentences. <|eot_id|>"
    sys_id = global_tokenizer(sys, add_special_tokens=False).input_ids

    context_id = sys_id

    biased_index = []
    num_demon = 20
    curren_position = len(sys_id)

    for idx in range(num_demon):

        demonstration = "<|start_header_id|>user<|end_header_id|>\n\n" + "Dialogue: " + samsum['train'][idx]['dialogue'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + "Summary: " + samsum['train'][idx]['summary'] + "<|eot_id|>"
        demonstration_id = global_tokenizer(demonstration, add_special_tokens=False).input_ids

        context_id += demonstration_id

        biased_index.append([curren_position, curren_position + len(demonstration_id)])

        curren_position += len(demonstration_id)

    new_ids, new_ranges = insert_mem_tokens(
        context_id, biased_index, compress_tokens, mem_start, mem_end
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

    total_num = len(samsum['test'])
    total_score = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))

        new_prompt = "<|start_header_id|>user<|end_header_id|>\n\n" + "Dialogue: " + samsum['test'][i]['dialogue'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nSummary: "
        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        generate_id = torch.cat([filtered_id, prompt_id], dim = 1)

        with torch.no_grad():

            outputs = global_model.generate(
                input_ids=generate_id,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=filtered_kv,
                use_cache=True
            )
        # print(outputs)
        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]

        score = calculate_rouge_l_score(response, samsum['test'][i]['summary'])

        total_score = total_score + score

        res_list.append({"id": str(i),"dialogue": samsum['test'][i]['dialogue'], "response": response, "gold_answer": samsum['test'][i]['summary'], "Score": score})
        print("Response", response)
        # print("Gold_ans", samsum['test'][i]['summary'])
        print("Score", score)

    avg_score = total_score / total_num
    print(avg_score)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/order/compress/Samsum_demon{num_demon}_ckpt{ckpt}_{avg_score}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
