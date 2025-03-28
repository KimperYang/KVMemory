import argparse
import datetime
import json

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.utils.do_blend import append_kv, do_blend


def calculate_rouge_l_score(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    # rouge_l_score = scores['rougeL'].recall
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score

def main():

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--weight', type=int, required=True, help='Run name')

    args = parser.parse_args()

    weight = args.weight

    model_name = f"meta-llama/Llama-3.2-{weight}B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to('cuda')

    multinews = load_dataset("alexfabbri/multi_news")

    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an AI assistant who summarizes the article. <|eot_id|>"

    num_demon = 3
    contexts = [sys]

    for idx in range(num_demon):

        demonstration = "<|start_header_id|>user<|end_header_id|>\n\n" + "Article: " + multinews['train'][idx]['document'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + "Summary: " + multinews['train'][idx]['summary'] + "<|eot_id|>"
        contexts.append(demonstration)

    start_pos = 0
    concat_ids = []
    kv_list = []
    current_pos = start_pos

    for st in contexts:
        input_ids = tokenizer(st, add_special_tokens=False).input_ids
        position_ids = list(range(current_pos, current_pos + len(input_ids)))
        with torch.no_grad():
            output = model(input_ids=torch.tensor([input_ids],device=model.device),
                        position_ids=torch.tensor([position_ids],device=model.device))
        kv_list.append(output.past_key_values)

        concat_ids += input_ids
        current_pos += len(input_ids)

    old_kv = append_kv(kv_list)

    global_position_ids = torch.tensor([list(range(start_pos, start_pos + len(concat_ids)))],device=model.device)
    with torch.no_grad():
        output = model(input_ids=torch.tensor([concat_ids],device=model.device),
                        position_ids=global_position_ids,
                        output_hidden_states=True)

    golden_kv = output.past_key_values
    first_layer_states = output.hidden_states[2]

    blend_kv = do_blend(model=model, old_kv=old_kv, golden_kv=golden_kv, recompute_ratio=0.18,first_layer_states=first_layer_states, position_ids=global_position_ids, config=config)

    total_num = 500
    total_score = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))

        new_prompt = "<|start_header_id|>user<|end_header_id|>\n\n" + "Article: " + multinews['validation'][i]['document'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nSummary: "
        prompt_id = tokenizer(new_prompt, add_special_tokens=False).input_ids

        generate_id = torch.tensor([concat_ids + prompt_id], device=model.device)

        with torch.no_grad():

            outputs = model.generate(
                input_ids=generate_id,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=blend_kv,
                use_cache=True
            )
        # print(outputs)
        generated_seq = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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

    file_name = f"result/new_data/cacheblend_{weight}B/multinews_demon{num_demon}_{avg_score}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
