import argparse
import datetime
import json
import string
from typing import List

import pandas as pd
import regex
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.utils.do_blend import append_kv, do_blend_filter


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
    parser.add_argument('--pos', type=int, required=True, help='Position value')

    args = parser.parse_args()

    pos = args.pos

    if pos in [0, 4, 9]:
        jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
    else:
        jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model.to('cuda')

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."

    total_num = 500
    correct_num = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))
        texts = [template]
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
            texts.append(f"Document [{j+1}](Title: {title}) {text}\n")

        start_pos = 0
        concat_ids = []
        kv_list = []
        current_pos = start_pos

        for st in texts:
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

        blend_kv = do_blend_filter(model=model, old_kv=old_kv, golden_kv=golden_kv, recompute_ratio=0.18, first_layer_states=first_layer_states, position_ids=global_position_ids, config=config)

        new_prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        prompt_id = tokenizer(new_prompt, add_special_tokens=False).input_ids

        generate_id = torch.tensor([concat_ids + prompt_id], device=model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=generate_id,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=blend_kv,
                use_cache=True
            )
        # print(outputs)
        generated_seq = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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

    file_name = f"result/llama31/cacheblend/NQ_at{pos}_ratio18_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
