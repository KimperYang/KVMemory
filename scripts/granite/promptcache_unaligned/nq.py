"""
PromptCache baseline for Granite on NQ, WITHOUT position-id alignment.

Same block-diagonal prefill as `scripts/granite/promptcache/nq.py`, but each
segment's `position_ids` restart from 0 (system at 0..L_sys-1, each doc at
0..L_doc-1, user question at 0..L_q-1). Generation tokens then use HF's
default positions continuing from the flat prefill length, so queries and
cached keys are no longer aligned in RoPE space.

```
python scripts/granite/promptcache_unaligned/nq.py --pos 0 --batch_size 1
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

parser = argparse.ArgumentParser(description="Granite PromptCache (unaligned positions) on NQ.")
parser.add_argument("--pos", type=int, required=True, help="Position value for the gold document.")
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


def preprocess_fn(example: Dict[str, str], tokenizer, target_position: int):
    system = "<|start_of_role|>system<|end_of_role|>You are a helpful, respectful and honest assistant.<|end_of_text|>\n"

    question = example["question"]
    doc_list = [example["ctxs"][x]["text"] for x in range(10)]
    title_list = [example["ctxs"][x]["title"] for x in range(10)]
    if target_position not in [0, 4, 9]:
        gt_doc = doc_list.pop(0)
        gt_title = title_list.pop(0)
        doc_list.insert(target_position, gt_doc)
        title_list.insert(target_position, gt_title)

    system_ids = tokenizer(system, add_special_tokens=False).input_ids
    input_ids = list(system_ids)
    segment_ids = [1] * len(system_ids)
    position_ids = list(range(len(system_ids)))

    for j in range(10):
        doc_str = f"Document [{j+1}](Title: {title_list[j]}) {doc_list[j]}\n"
        doc_ids = tokenizer(doc_str, add_special_tokens=False).input_ids
        input_ids += doc_ids
        segment_ids += [j + 2] * len(doc_ids)
        position_ids += list(range(len(doc_ids)))  # restart from 0 for each doc

    user_block = (
        "<|start_of_role|>user<|end_of_role|>" + question + "<|end_of_text|>"
    )
    user_ids = tokenizer(user_block, add_special_tokens=False).input_ids
    input_ids += user_ids
    segment_ids += [12] * len(user_ids)
    position_ids += list(range(len(user_ids)))  # restart from 0 for the user question

    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "position_ids": position_ids,
    }


class DataCollatorForGeneration():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids = []
        segment_ids = []
        position_ids = []
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
            position_ids.append(item["position_ids"] + [0] * residual)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "segment_ids": torch.LongTensor(segment_ids),
            "position_ids": torch.LongTensor(position_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def main():
    pos = args.pos
    batch_size = args.batch_size
    device = torch.device("cuda")

    if pos in [0, 4, 9]:
        data_path = f"data/raw/nq/nq-open-10_{pos}.jsonl"
    else:
        data_path = "data/raw/nq/nq-open-10_0.jsonl"
    dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    print(dataset)
    all_answers = dataset["answers"]
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
        fn_kwargs=dict(tokenizer=tokenizer, target_position=pos),
    )

    total_num = 500
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
        position_ids = batch["position_ids"]
        attention_mask_for_pad = batch["attention_mask"]

        with torch.no_grad():
            input_ids = move_to_target_device(input_ids, device)
            attention_mask_4d = move_to_target_device(attention_mask_4d, device)
            attention_mask_for_pad = move_to_target_device(attention_mask_for_pad, device)
            position_ids = move_to_target_device(position_ids, device)

            prefilling_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
            )
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

        scores = [best_subspan_em(responses[idx], batch_answers[idx]) for idx in range(curr_batch_size)]
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

    file_name = f"result/promptcache_unaligned_NQ_at{pos}_{accuracy}_{time_str}.jsonl"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")

    print(f"Dumped at {file_name}")


if __name__ == "__main__":
    main()
