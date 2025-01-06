import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
import string
from typing import List
from src.data.attention import construct_biased_attention_matrix
import regex

jsonObj = pd.read_json(path_or_buf='/mnt/data2/jingbo/2wiki/kv_dump_bias_30000steps_bsz256_5e-6_full_ckpt16000_0.9968988549618321_20241120-131044.jsonl', lines=True)

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

    correct_num = 0
    res_list = []

    for i in range(len(jsonObj)):

        # print("Processing sample:", str(i))

        response = jsonObj['response'][i]
        answer = [jsonObj['gold_answer'][i]]
        score = best_subspan_em(response, answer)

        correct_num = correct_num + int(score)

        
    accuracy = correct_num / len(jsonObj)
    print("Acc:", accuracy)

if __name__ == "__main__":
    main()
