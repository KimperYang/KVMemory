import pandas as pd   
from rouge_score import rouge_scorer

def compute_rouge_l_recall(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)  # The order is reference first, predicted second
    recall = scores['rougeL'].recall
    return recall

# file_path = "/home/jingbo/KVMemory/result/dialog/dialog_finetune_0.059485831506524464_20240926-050748.jsonl"
file_path = "/home/jingbo/KVMemory/result/dialog/dialog_baseline_longer_0.06989317395065889_20240925-210824.jsonl"
jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
res = []

for i in range(len(jsonObj)):
    # print(i)
    predicted = jsonObj["response"][i]
    reference = jsonObj["gold_answer"][i]
    recall_rate = compute_rouge_l_recall(predicted, reference)
    res.append(recall_rate)
    # print(recall_rate)


print(sum(res)/len(res))