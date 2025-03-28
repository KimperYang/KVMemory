import argparse
import datetime
import json

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.attention import construct_biased_attention_matrix


def calculate_rouge_l_score(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score

def main():
    special_token_start = 128011
    mem_start = 128054
    mem_end = 128055

    parser = argparse.ArgumentParser(description="Run script with specified clinic and pos.")
    parser.add_argument('--run', type=str, required=True, help='Run name')
    parser.add_argument('--reencode', type=int, required=True, help='Checkpoint number')

    args = parser.parse_args()

    run_name = args.run
    reencode_num = args.reencode

    global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-6000")

    global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-6000", torch_dtype=torch.bfloat16)
    global_model.to('cuda')

    multinews = load_dataset("alexfabbri/multi_news", trust_remote_code=True)

    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant. Summarize the text given below in detail.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

    # total_num = len(multinews['validation'])
    total_num = 500
    total_score = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))
        sys_id = global_tokenizer(sys, add_special_tokens=False).input_ids
        sys_id += [mem_start]

        input_ids = sys_id

        document_id = global_tokenizer(multinews['validation'][i]['document'], add_special_tokens=False).input_ids

        if len(document_id) > 4096:
            continue

        chunks = [document_id[i:i+100] for i in range(0, len(document_id), 100)]

        biased_index = []
        current_index = len(sys_id)

        for j in range(len(chunks)):

            tem_id = chunks[j]

            for sub_idx in range(reencode_num):
                tem_id = tem_id + [special_token_start + reencode_num * j + sub_idx]

            biased_index.append([current_index, current_index + len(tem_id) - reencode_num])

            current_index += len(tem_id)

            input_ids += tem_id

        input_ids += [mem_end]

        new_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = global_tokenizer(new_prompt, add_special_tokens=False).input_ids

        attention_matrix  = construct_biased_attention_matrix(len(input_ids), biased_index, len(input_ids), global_model.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            prefill = global_model(input_ids = torch.tensor([input_ids],device=global_model.device),
                                   attention_mask = attention_matrix)

            outputs = global_model.generate(
                input_ids=torch.tensor([input_ids + prompt_id], device=global_model.device),
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=prefill.past_key_values,
                use_cache=True
            )

        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]

        score = calculate_rouge_l_score(response, multinews['validation'][i]['summary'])

        total_score = total_score + score

        res_list.append({"id": str(i),"dialogue": multinews['validation'][i]['document'], "response": response, "gold_answer": multinews['validation'][i]['summary'], "Score": score})
        print("Response", response)
        print("Score", score)

    avg_score = total_score / total_num
    print(avg_score)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/{run_name}/multinews_chunk_{avg_score}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
