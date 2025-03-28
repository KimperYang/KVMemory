import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd    
import json
import datetime
import string
from typing import List
from src.utils.do_blend import append_kv, do_blend
import regex
import argparse
from datasets import load_dataset
# vocab_size = len(global_tokenizer)
# base_model.resize_token_embeddings(vocab_size)

# peft_config_path = "/mnt/data/jingbo/kv_dump_combine_mix5_5000steps_5e-6_full/checkpoint-5000"  # Path to the directory where LoRA weights are stored

# global_model = PeftModel.from_pretrained(base_model, peft_config_path)

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
    parser.add_argument('--weight', type=int, required=True, help='Checkpoint number')
    parser.add_argument('--run', type=str, required=True, help='Checkpoint number')

    args = parser.parse_args()

    run_name = args.run
    weight = args.weight

    data_list=load_dataset("dgslibisey/MuSiQue", split='validation')

    tokenizer = AutoTokenizer.from_pretrained(run_name)
    model = AutoModelForCausalLM.from_pretrained(run_name, torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained(run_name)
    model.to('cuda')

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

    total_num = len(data_list)
    correct_num = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))
        memory_list = [template]

        for j in range(len(data_list[i]['paragraphs'])):
            title = data_list[i]['paragraphs'][j]['title']
            text = data_list[i]['paragraphs'][j]['paragraph_text']
            memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

        start_pos = 0
        concat_ids = []
        kv_list = []
        current_pos = start_pos

        for st in memory_list:
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

        new_prompt = data_list[i]['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = tokenizer(new_prompt, add_special_tokens=False).input_ids

        model.eval()

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
        print("response:", response)

        score = best_subspan_em(response, [data_list[i]['answer']])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data_list[i]['question'], "response": response, "gold_answer": [data_list[i]['answer']], "Score": score})
        print("Correct progress", correct_num)

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/new_data/cacheblend_{weight}B/musique_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
