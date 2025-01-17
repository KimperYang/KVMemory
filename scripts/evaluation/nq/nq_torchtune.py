"""
Evaluate the performance of LLMs fine-tuned via TorchTitan on NQ dataset.

1. Convert the Titan checkpoint from DCP to torch:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
    torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

2. Run evaluation:
```
python scripts/evaluation/nq/nq_torchtune.py \
    --ckpt_path checkpoint.pt \
    --pos 1
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
from safetensors import safe_open
from torch.utils.data import DataLoader
from torchtune.models.convert_weights import tune_to_hf
from transformers import LlamaForCausalLM

from src.common import move_to_target_device
from src.data.attention import construct_biased_attention_matrix
from src.model.titan_preprocessor import LLaMA32Tokenizer

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    required=True,
    help="The path to the `checkpoint.pt` file.",
)
parser.add_argument("--pos", type=int, required=True, help="Position value")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the evaluation.")

args = parser.parse_args()

def load_model_weights(ckpt_path: str):
    safe_tensor_file = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(safe_tensor_file):
        state_dict = {}
        with safe_open(safe_tensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        # state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
        return state_dict

    state_dict = torch.load(ckpt_path, weights_only=False)

    state_dict = state_dict["model"]
    state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

    converted_state_dict = tune_to_hf(
        state_dict=state_dict,
        num_heads=32,
        num_kv_heads=8,
        dim=2048,
    )
    return converted_state_dict

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

def preprocess_fn(example: Dict[str, str], tokenizer: LLaMA32Tokenizer, target_position: int):
    template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, "
        "respectful and honest assistant.<|eot_id|>"
    )
    question = example["question"]
    memory_list = []
    for j in range(0,10):
        title = example["ctxs"][j]["title"]
        text = example["ctxs"][j]["text"]
        memory_list.append("<|reserved_special_token_3|>" + f"Document [{j+1}](Title: {title}) {text}" + "\n<|reserved_special_token_4|>")
    # If target_position == 1, for example, then the code will read from the data source that
    # always put groud-truth at position 0.
    if target_position not in [0, 4, 9]:
        ground_truth = memory_list.pop(0)
        memory_list.insert(target_position, ground_truth)

    memory_list.insert(0, template)
    biased_index = []
    id_list = []

    idx = 0

    for st in memory_list:

        # tem_id = tokenizer(st, return_tensors="pt", add_special_tokens=False).input_ids
        tem_id = tokenizer(st, allowed_special="all", disallowed_special = None, add_special_tokens = False)["input_ids"]
        biased_index.append([idx, idx + len(tem_id)])

        id_list += tem_id
        idx = idx + len(tem_id)

    new_prompt = (
        "<|reserved_special_token_5|><|start_header_id|>user<|end_header_id|>\n\n"
        "Write a high-quality answer for the given question using only the provided "
        f"search results (some of which might be irrelevant). Question: {question}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    prompt_id = tokenizer(new_prompt, allowed_special="all", disallowed_special = None, add_special_tokens = False)["input_ids"]
    input_ids = id_list + prompt_id
    return {
        "input_ids": input_ids,
        "biased_index": biased_index,
    }

class DataCollatorForGeneration():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        pass
    def __call__(self, batch):
        input_ids = []
        biased_index = []
        mem_num = []
        input_length = []
        attention_mask = []
        for item in batch:
            if item['biased_index'] is not None:
                mem_num.append(len(item['biased_index']))
            else:
                mem_num.append(0)
            input_length.append(len(item['input_ids']))

        max_mem_num = max(mem_num)
        max_length = max(input_length)

        for item in batch:
            seq_length = len(item['input_ids'])
            _mem_num = len(item['biased_index']) if item['biased_index'] is not None else 0

            residual = max_length - seq_length
            padded_input_ids = [self.pad_id] * residual + item['input_ids']
            curr_attention_mask = [0] * residual + [1] * seq_length


            original_biased_index = item['biased_index']
            converted_biased_index = []
            for start_stop_pair in original_biased_index:
                start = start_stop_pair[0]
                stop = start_stop_pair[1]
                converted_biased_index.append([start + residual, stop + residual])

            converted_biased_index = converted_biased_index + [[0,0]] * (max_mem_num - _mem_num)
            input_ids.append(padded_input_ids)
            attention_mask.append(curr_attention_mask)
            biased_index.append(converted_biased_index)

        return {
            'input_ids': torch.LongTensor(input_ids),
            'biased_index': torch.LongTensor(biased_index),
            "input_length": torch.LongTensor(input_length),
            'mem_num': torch.LongTensor(mem_num),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def main():
    ckpt_path = args.ckpt_path
    pos = args.pos
    batch_size: int = args.batch_size
    device = torch.device("cuda")

    if pos in [0, 4, 9]:
        data_path = f"data/raw/nq/nq-open-10_{pos}.jsonl"
    else:
        data_path = "data/raw/nq/nq-open-10_0.jsonl"
    dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    print(dataset)
    all_answers = dataset["answers"]
    print(all_answers[:10])

    tokenizer = LLaMA32Tokenizer(model_path="data/titan_tokenizer/original/tokenizer.model")
    state_dict = load_model_weights(ckpt_path)

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16,
    )
    # model.load_state_dict(state_dict, strict=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    exist_columns = dataset.column_names
    dataset = dataset.map(
        preprocess_fn,
        batched=False,
        num_proc=16,
        remove_columns=exist_columns,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            target_position=pos,
        ),
    )

    total_num = 500
    dataset = dataset.select(np.arange(total_num))
    correct_num = 0
    res_list = []

    collate_fn = DataCollatorForGeneration(pad_id=tokenizer.pad_id)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for batch_id, batch in enumerate(eval_dataloader):
        curr_batch_size = batch['input_ids'].size(0)
        batch_answers = all_answers[batch_id * batch_size : batch_id * batch_size + curr_batch_size]
        attention_matrices = []
        max_length = max(batch['input_length'])
        for idx in range(len(batch['input_ids'])):
            mem_num = batch['mem_num'][idx]
            if mem_num == 0:
                biased_ranges = None
            else:
                biased_ranges = batch['biased_index'][idx][:mem_num]
            # (1, 1, input_length, input_length)
            block_attenntion_mask = construct_biased_attention_matrix(
                batch['input_length'][idx],
                biased_ranges,
                max_length,
                batch['input_ids'].device
            ).unsqueeze(0)
            pad_mask = batch["attention_mask"]
            pad_length = torch.sum(pad_mask == 0)
            block_attenntion_mask[:, :, :pad_length] = float('-inf')

            attention_matrices.append(block_attenntion_mask)

        attention_mask = torch.stack(attention_matrices)
        input_ids = batch["input_ids"]

        with torch.no_grad():
            input_ids = move_to_target_device(input_ids, device)
            attention_mask = move_to_target_device(attention_mask, device)
            # outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            # past_key_values = outputs.past_key_values

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                do_sample=False,
                # past_key_values=past_key_values,
                use_cache=True
            )
        generated_seqs = [tokenizer.decode(
                outputs[i, input_ids.size(1):],
            )
            for i in range(input_ids.size(0))
        ]

        print(generated_seqs)
        response = [
            generated_seq.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]
            for generated_seq in generated_seqs
        ]
        responses = generated_seqs
        print(responses)

        scores = [best_subspan_em(responses[idx], batch_answers[idx]) for idx in range(curr_batch_size)]
        for idx, score in scores:
            correct_num = correct_num + int(score)
            res_list.append(
                {
                    # "question": question,
                    "response": responses[idx],
                    "gold_answer": batch_answers[idx],
                    "score": scores[idx],
                }
            )
        print("Correct progress", correct_num)

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/decay/NQ_at{pos}_{accuracy}_{time_str}.jsonl"

    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()

