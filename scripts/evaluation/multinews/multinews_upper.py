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
    # rouge_l_score = scores['rougeL'].recall
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score

def main():

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--run', type=str, required=True, help='Run name')
    parser.add_argument('--weight', type=int, required=False, help='Run name')

    args = parser.parse_args()

    run_name = args.run
    weight = args.weight

    if "meta" in run_name:
        global_tokenizer = AutoTokenizer.from_pretrained(run_name)
        global_model = AutoModelForCausalLM.from_pretrained(run_name, torch_dtype=torch.bfloat16)
    else: 
        global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-6000")
        global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-6000", torch_dtype=torch.bfloat16)

    global_model.to('cuda')

    multinews = load_dataset("alexfabbri/multi_news")

    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an AI assistant who summarizes the article. <|eot_id|>"
    sys_id = global_tokenizer(sys, add_special_tokens=False).input_ids
    context_id = sys_id

    num_demon = 2

    for idx in range(num_demon):

        demonstration = "<|start_header_id|>user<|end_header_id|>\n\n" + "Article: " + multinews['train'][idx]['document'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + "Summary: " + multinews['train'][idx]['summary'] + "<|eot_id|>"
        demonstration_id = global_tokenizer(demonstration, add_special_tokens=False).input_ids

        context_id += demonstration_id

    global_model.eval()
    with torch.no_grad():
            outputs = global_model(input_ids = torch.tensor([context_id],device=global_model.device))
            past_key_values = outputs.past_key_values

    total_num = 500
    total_score = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))

        new_prompt = "<|start_header_id|>user<|end_header_id|>\n\n" + "Article: " + multinews['validation'][i]['document'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nSummary: "
        prompt_id = global_tokenizer(new_prompt, add_special_tokens=False).input_ids

        generate_id = torch.tensor([context_id + prompt_id], device=global_model.device)

        with torch.no_grad():

            outputs = global_model.generate(
                input_ids=generate_id,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=past_key_values,
                use_cache=True
            )
        # print(outputs)
        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]

        score = calculate_rouge_l_score(response, multinews['validation'][i]['summary'])

        total_score = total_score + score

        res_list.append({"id": str(i),"dialogue": multinews['validation'][i]['document'], "response": response, "gold_answer": multinews['validation'][i]['summary'], "Score": score})
        print("Response", response)
        # print("Gold_ans", multinews['validation'][i]['summary'])
        print("Score", score)

    avg_score = total_score / total_num
    print(avg_score)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    if "meta" in run_name:
        file_name = f"result/new_data/original_{weight}B/multinews_500_original_demon{num_demon}_{avg_score}_{time_str}.jsonl"
    else:
        file_name = f"result/{run_name}/multinews_500_demon{num_demon}_{avg_score}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
