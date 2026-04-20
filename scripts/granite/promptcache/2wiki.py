"""
PromptCache baseline for Granite on 2WikiMultihopQA.

Training-free: loads the base `ibm-granite/granite-4.1-8b`, no fine-tuned
checkpoint, no added special tokens. System / each of 10 documents / user
question are all separate block-diagonal segments at prefill time.

```
python scripts/granite/promptcache/2wiki.py --batch_size 1
```
"""
import argparse
import datetime
import json
import os
import string
from typing import Dict, List

import datasets
import numpy as np
import regex
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from src.common import move_to_target_device
from src.data.titan_preprocessor import make_segment_mask

parser = argparse.ArgumentParser(description="Granite PromptCache baseline on 2WikiMultihopQA.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the evaluation.")
args = parser.parse_args()


def normalize_answer(s: str) -> str:
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


def preprocess_fn(example: Dict[str, str], tokenizer):
    system = "<|start_of_role|>system<|end_of_role|>You are a helpful, respectful and honest assistant.<|end_of_text|>\n"

    question = example["question"]

    system_ids = tokenizer(system, add_special_tokens=False).input_ids
    input_ids = list(system_ids)
    segment_ids = [1] * len(system_ids)

    num_docs = len(example['context'])
    for j in range(num_docs):
        title = example['context'][j][0]
        text = " ".join(example['context'][j][1])
        doc_str = f"Document [{j+1}](Title: {title}) {text}\n"
        doc_ids = tokenizer(doc_str, add_special_tokens=False).input_ids
        input_ids += doc_ids
        segment_ids += [j + 2] * len(doc_ids)

    user_block = (
        "<|start_of_role|>user<|end_of_role|>" + question + "<|end_of_text|>"
    )
    user_ids = tokenizer(user_block, add_special_tokens=False).input_ids
    input_ids += user_ids
    # segment id for question sits right after the last document segment
    segment_ids += [num_docs + 2] * len(user_ids)

    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
    }


class DataCollatorForGeneration():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids = []
        segment_ids = []
        attention_mask = []
        length_list = [len(x['input_ids']) for x in batch]
        max_length = max(length_list)

        for item in batch:
            seq_length = len(item['input_ids'])
            residual = max_length - seq_length
            padded_input_ids = item['input_ids'] + [self.pad_id] * residual
            curr_attention_mask = [1] * seq_length + [0] * residual
            input_ids.append(padded_input_ids)
            attention_mask.append(curr_attention_mask)
            segment_ids.append(item["segment_ids"] + [-1] * residual)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "segment_ids": torch.LongTensor(segment_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def main():
    batch_size = args.batch_size
    device = torch.device("cuda")

    data_path = "2WikiMultihopQA/dev.json"
    dataset = datasets.load_dataset("json", data_files=data_path)
    print(dataset)
    all_answers = dataset["answer"]
    print(all_answers[:10])

    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-8b")
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-4.1-8b",
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()

    exist_columns = dataset.column_names
    dataset = dataset.map(
        preprocess_fn,
        batched=False,
        num_proc=16,
        remove_columns=exist_columns,
        fn_kwargs=dict(tokenizer=tokenizer),
    )

    total_num = len(dataset)
    dataset = dataset.select(np.arange(total_num))
    correct_num = 0
    res_list = []

    collate_fn = DataCollatorForGeneration(pad_id=tokenizer.pad_token_id)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    prog_bar = auto_tqdm(range(len(eval_dataloader)))

    eot_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    generation_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=200,
        stop_strings=["<|end_of_text|>"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eot_id,
    )
    generation_prompt = "\n<|start_of_role|>assistant<|end_of_role|>"

    generation_token_ids = tokenizer(generation_prompt, add_special_tokens=False)["input_ids"]
    generation_token_ids = torch.LongTensor(generation_token_ids)
    generation_token_ids: torch.LongTensor = move_to_target_device(generation_token_ids, device)

    for batch_id, batch in enumerate(eval_dataloader):
        curr_batch_size = batch['input_ids'].size(0)
        batch_answers = all_answers[batch_id * batch_size : batch_id * batch_size + curr_batch_size]
        segment_ids = batch["segment_ids"]
        attention_mask = make_segment_mask(
            source_segments=segment_ids,
            target_segments=segment_ids,
            add_causal_lm_mask=True,
        )
        attention_mask_4d = attention_mask.unsqueeze(1)
        input_ids = batch["input_ids"]
        attention_mask_for_pad = batch["attention_mask"]

        with torch.no_grad():
            input_ids = move_to_target_device(input_ids, device)
            attention_mask_4d = move_to_target_device(attention_mask_4d, device)
            attention_mask_for_pad = move_to_target_device(attention_mask_for_pad, device)

            prefilling_outputs = model(input_ids=input_ids, attention_mask=attention_mask_4d)
            past_key_values = prefilling_outputs.past_key_values

            generation_prefix = generation_token_ids.repeat(curr_batch_size, 1)
            generation_input_ids = torch.cat([input_ids, generation_prefix], axis=1)
            attention_mask_for_pad = torch.cat(
                [attention_mask_for_pad, torch.ones_like(generation_prefix)], axis=1
            )
            outputs = model.generate(
                input_ids=generation_input_ids,
                attention_mask=attention_mask_for_pad,
                use_cache=True,
                generation_config=generation_cfg,
                past_key_values=past_key_values,
                tokenizer=tokenizer,
            )
        generated_seqs = [
            tokenizer.decode(outputs[i, input_ids.size(1):].tolist())
            for i in range(input_ids.size(0))
        ]

        responses = [
            generated_seq.split("<|start_of_role|>assistant<|end_of_role|>")[-1].strip().split("<|end_of_text|>")[0]
            for generated_seq in generated_seqs
        ]
        for idx, x in enumerate(responses):
            print(x)
            print("Ground-truth: ", batch_answers[idx])
            print("------\n")

        scores = [best_subspan_em(responses[idx], [batch_answers[idx]]) for idx in range(curr_batch_size)]
        for idx, score in enumerate(scores):
            correct_num = correct_num + int(score)
            res_list.append(
                {
                    "response": responses[idx],
                    "gold_answer": batch_answers[idx],
                    "score": scores[idx],
                }
            )
        print("Correct progress", correct_num)
        prog_bar.update(1)

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/promptcache_wiki_{accuracy}_{time_str}.jsonl"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")

    print(f"Dumped at {file_name}")


if __name__ == "__main__":
    main()
