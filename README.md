# KVMemory

Train and evaluate long-context language models with memory-token attention. This README covers the end-to-end pipeline for the Granite 4.1 8B line of experiments: data preprocessing → training (three settings) → evaluation.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

The project is pinned against `transformers==4.43.x`. DeepSpeed is required for the 8×H100 config (`configs/h100_config.yaml`); FSDP users can skip it if they only launch with `configs/fsdp.yaml`.

## Training settings

Three Granite training entry points live at the repo root, each corresponding to a different attention/memory configuration:

| Setting | Entry point | Preprocessor | Custom trainer | `reencode_num` | Memory tokens |
|---|---|---|---|---|---|
| Upperbound (standard attention) | `granite_upperbound_trainer.py` | `granite_baseline_attention_preprocessor` | `Trainer` | — | none |
| KVLink-0 (blocked attention, no link tokens) | `granite_kvlink0_trainer.py` | `granite_sum_attention_preprocessor` | `CustomTrainerBiasAttn` | 0 | `<mem_start>`, `<mem_end>` |
| KVLink-5 (blocked attention with 5 link tokens per chunk) | `granite_kvlink5_trainer.py` | `granite_sum_attention_preprocessor` | `CustomTrainerBiasAttn` | 5 | `<mem_start>`, `<mem_end>`, `<link_0>` … `<link_199>` |

All three settings share the same data mix (FineWeb text, Tulu SFT, DaringAnteater SFT with memory, Block-QA with/without memory, XSum) and the same Granite chat template (`<|start_of_role|>…<|end_of_role|>…<|end_of_text|>\n`).

## 1. Data processing

### 1.1 Download raw sources

| Dataset | Where to get it | Destination |
|---|---|---|
| FineWeb | pulled automatically by `fineweb.py` via `datasets.load_dataset` | — |
| Tulu | pulled automatically by `tulu.py` | — |
| Daring-Anteater | pulled automatically by `daring_anteater.py` from `nvidia/Daring-Anteater` | — |
| XSum | pulled automatically by `sum.py` | — |
| Block-QA | manual download: [block_qa.zip](https://drive.google.com/file/d/1wjSX2C0OzvmWY18JL76Y0VLj-WiMIA_6/view?usp=sharing) | `data/raw/block_qa/block_qa.jsonl` |
| NQ | manual (`nq-open-10_{pos}.jsonl`) | `data/raw/nq/` |
| 2WikiMultihopQA | manual (`dev.json`) | `2WikiMultihopQA/dev.json` |
| HotpotQA (distractor) | pulled automatically by the eval script | — |
| MuSiQue | pulled automatically by the eval script | — |
| TriviaQA | manual download: [tqa.zip](https://drive.google.com/file/d/1wnIZGQo3vMrVH9AQ8_Lnhzkk1bpkLuJD/view?usp=sharing) | `data/raw/tqa/eval.jsonl` |

Set up the Block-QA source first:

```bash
unzip block_qa.zip
mkdir -p data/raw/block_qa
mv block_qa.jsonl data/raw/block_qa
```

### 1.2 Preprocess into `dataset_cache/processed/`

Run each preprocessing script from the repo root. Each one writes a `DatasetDict` with `train`/`test` splits to disk under `dataset_cache/processed/`:

```bash
python scripts/data_process/fineweb.py         --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
python scripts/data_process/tulu.py            --max_length=4096 --validation_size=2000
python scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000
python scripts/data_process/QA.py              --max_length=4096 --validation_size=2000
python scripts/data_process/sum.py             --max_length=4096 --validation_size=1000
```

The Granite trainers read from these paths at runtime:

```
dataset_cache/processed/fineweb/text            # pre-training text
dataset_cache/processed/tulu/sft                # Tulu SFT (no memory)
dataset_cache/processed/daringanteater/sft_mem  # SFT with memory turns
dataset_cache/processed/block_qa/qa             # QA (no memory)
dataset_cache/processed/block_qa/qa_mem         # QA with memory
dataset_cache/processed/xsum/xsum               # XSum summarization
```

The actual Granite chat-template tokenization happens **inside** the preprocessor classes in `src/data/input_preprocessor.py` (`granite_baseline_attention_preprocessor` for upperbound, `granite_sum_attention_preprocessor` for kvlink). The on-disk caches are raw JSON-like structures; tokenization is applied on-the-fly via `dataset.map(...)` in the trainers, so **you only need to reprocess if the raw data changes**, not when you switch between upperbound / kvlink0 / kvlink5.

## 2. Training

### 2.1 Accelerate configs

- `configs/single_gpu.yaml` — single-GPU debug runs
- `configs/h100_config.yaml` — single-node DeepSpeed Zero-2, 8×H100
- `configs/h100x6_config.yaml` — 6-GPU variant
- `configs/fsdp.yaml` — FSDP-based multi-node / memory-constrained runs

### 2.2 Upperbound (standard attention, no memory tokens)

Trains Granite with the full context attending causally — this is the reference ceiling.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/h100_config.yaml \
    --main_process_port 25678 \
    granite_upperbound_trainer.py
```

Key hyperparameters (edit the file to change): `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `max_steps=6000`, `lr=5e-6`, cosine schedule with 10% warmup, bf16, gradient checkpointing. Output: `training_res/upperbound_granite_8B/`.

### 2.3 KVLink-0 (blocked attention, no link tokens)

Introduces `<mem_start>` and `<mem_end>` boundaries and uses blocked attention (`CustomTrainerBiasAttn` with `custom_collate_bias`). Each memory chunk is a separate block that does not attend to other chunks during prefill.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/h100_config.yaml \
    --main_process_port 25678 \
    granite_kvlink0_trainer.py
```

Defaults: `reencode_num=0`, `max_memory_num=40` ⇒ only 2 extra tokens added to the vocab (`<mem_start>`, `<mem_end>`). Output: `training_res/kvlink_0_granite_8B/`.

### 2.4 KVLink-5 (blocked attention with 5 link tokens per chunk)

Same as KVLink-0 but each memory chunk gets 5 trainable link tokens (`<link_j*5+i>` for `i in range(5)`), giving the model cross-chunk connective tissue.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/h100_config.yaml \
    --main_process_port 25678 \
    granite_kvlink5_trainer.py
```

Defaults: `reencode_num=5`, `max_memory_num=40` ⇒ 200 link tokens + `<mem_start>` + `<mem_end>` = 202 new tokens. `model.resize_token_embeddings` is called to grow the output head accordingly. Output: `training_res/kvlink_5_granite_8B/`.

### 2.5 Notes shared by all three trainers

- Base checkpoint: `ibm-granite/granite-4.1-8b` (loaded via `AutoModelForCausalLM.from_pretrained`).
- Attention backend: `sdpa` (change to `flash_attention_2` by editing the `attn_implementation` argument).
- WANDB: project `kvmemory`; set `WANDB_API_KEY` before launching.
- Checkpoints are HuggingFace-format (`model.safetensors` + tokenizer + config), saved every `save_steps` under `training_res/…/checkpoint-XXXX/`.

## 3. Evaluation

The four Granite eval scripts live under `scripts/granite/` and share the same CLI shape. They assume HuggingFace-format checkpoints (use `--hf True`).

### 3.1 Common arguments

```
--ckpt_path  path to the training checkpoint directory (e.g. training_res/kvlink_5_granite_8B/checkpoint-6000)
--batch_size per-GPU batch size for evaluation
--attn_type  "blocked" (for kvlink*) or "standard" (for upperbound)
--reencode_num  0 for upperbound / kvlink0, 5 for kvlink5
--hf         True if loading an HF-format checkpoint (recommended for Granite)
```

The script rebuilds the tokenizer state exactly as training did: it loads the base Granite tokenizer, appends `<link_*>` (length = `max_memory_num * reencode_num` = `40 * reencode_num`), then `<mem_start>`, `<mem_end>`. `special_token_start`, `mem_start`, `mem_end` are derived from `len(tokenizer)` so they stay in lockstep with training.

### 3.2 Natural Questions (NQ)

`--pos` selects where to place the gold document among the 10 distractors (0, 4, 9 use pre-shuffled files; others re-insert into slot 0's file).

```bash
# kvlink-5
for pos in 0 1 2 3 4 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=0 python scripts/granite/nq.py \
        --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
        --pos $pos --batch_size 4 \
        --attn_type blocked --reencode_num 5 --hf True
done

# upperbound
CUDA_VISIBLE_DEVICES=0 python scripts/granite/nq.py \
    --ckpt_path training_res/upperbound_granite_8B/checkpoint-6000 \
    --pos 0 --batch_size 4 \
    --attn_type standard --reencode_num 0 --hf True
```

Requires `data/raw/nq/nq-open-10_{0,4,9}.jsonl`. Results are written to `result/NQ_at{pos}_{acc}_{timestamp}.jsonl`.

### 3.3 2WikiMultihopQA

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/granite/2wiki.py \
    --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
    --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
```

Requires `2WikiMultihopQA/dev.json` at the repo root. Output: `result/wiki_{acc}_{timestamp}.jsonl`.

### 3.4 HotpotQA (distractor)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/granite/hqa.py \
    --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
    --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
```

Dataset is pulled automatically from `hotpotqa/hotpot_qa`. Output: `result/hqa_{acc}_{timestamp}.jsonl`.

### 3.5 MuSiQue

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/granite/musique.py \
    --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
    --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
```

Dataset is pulled automatically from `dgslibisey/MuSiQue`. Output: `result/musique_{acc}_{timestamp}.jsonl`.

### 3.6 Trivia QA

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/granite/tqa.py \
    --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
    --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
```

Requires the manually downloaded JSONL at `data/raw/tqa/eval.jsonl` (each row has `question`, `answers`, and 10 `documents` with `title`/`text`). Dataset is available [here](https://drive.google.com/file/d/1wnIZGQo3vMrVH9AQ8_Lnhzkk1bpkLuJD/view?usp=sharing). Output: `result/tqa_{acc}_{timestamp}.jsonl`.

### 3.7 Which `--attn_type` / `--reencode_num` to pair with which checkpoint

| Checkpoint source | `--attn_type` | `--reencode_num` |
|---|---|---|
| `granite_upperbound_trainer.py` | `standard` | `0` |
| `granite_kvlink0_trainer.py` | `blocked` | `0` |
| `granite_kvlink5_trainer.py` | `blocked` | `5` |

Using `standard` with a kvlink checkpoint (or vice versa) will produce degraded metrics because the memory boundaries won't match what the model saw during training.

## 4. Prefill latency timing

`scripts/timer/` contains micro-benchmarks that measure the wall-clock cost of a **single prefill pass**. They operate on random token ids (no real accuracy is computed), so they only require the weights and tokenizer — no data preprocessing needed.

Two Granite scripts:

| Script | What it times |
|---|---|
| `scripts/timer/granite_baseline.py` | Upperbound path — flat prefill over `[sys | batch_size × sequence_length tokens | user]`. |
| `scripts/timer/granite_sum.py` | KVLink path — move pre-cached per-chunk KVs back onto GPU, concat, re-apply RoPE with global positions, then prefill `[sys | <mem_start> | link tokens | <mem_end> | user]` against the concatenated cache with a kvlink-style 4-D attention mask. |

Both scripts default to 10 warm-up iterations + 100 timed iterations and print per-iteration time plus an average. The `sum` script uses `model.model.rotary_emb` so it is architecture-agnostic (works for any model that exposes that attribute).

### 4.1 Upperbound timing

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/timer/granite_baseline.py \
    --ckpt_path training_res/upperbound_granite_8B/checkpoint-6000 \
    --batch_size 10 --sequence_length 500
```

Any Granite HF checkpoint works (including the base `ibm-granite/granite-4.1-8b` if you just want to benchmark the untuned model — pass that as `--ckpt_path`).

### 4.2 KVLink timing

```bash
# kvlink-5
CUDA_VISIBLE_DEVICES=0 python scripts/timer/granite_sum.py \
    --ckpt_path training_res/kvlink_5_granite_8B/checkpoint-6000 \
    --reencode_num 5 --batch_size 10 --sequence_length 500

# kvlink-0 (no link tokens; <mem_start>/<mem_end> still used)
CUDA_VISIBLE_DEVICES=0 python scripts/timer/granite_sum.py \
    --ckpt_path training_res/kvlink_0_granite_8B/checkpoint-6000 \
    --reencode_num 0 --batch_size 10 --sequence_length 500
```

The checkpoint must carry the tokenizer with the extra special tokens (`<mem_start>`, `<mem_end>`, and `<link_*>` when `reencode_num > 0`). `AutoTokenizer.from_pretrained(ckpt_path)` picks these up automatically because `Trainer` saved them alongside the model weights. Pair `--reencode_num` with the setting the checkpoint was trained on — `0` for kvlink-0, `5` for kvlink-5.

### 4.3 Tuning knobs

| Flag | Default | Effect |
|---|---|---|
| `--batch_size` | `10` | Number of cached document chunks — total cached KV length = `batch_size × sequence_length`. |
| `--sequence_length` | `500` | Tokens per chunk. |
| `--warmup` | `10` | Iterations excluded from the reported average. |
| `--iters` | `110` | Total iterations (warm-up + timed). |

## 5. End-to-end quick start (kvlink-5)

```bash
# 1. Preprocess data (once)
unzip block_qa.zip && mkdir -p data/raw/block_qa && mv block_qa.jsonl data/raw/block_qa
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
python scripts/data_process/tulu.py            --max_length=4096 --validation_size=2000
python scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000
python scripts/data_process/QA.py              --max_length=4096 --validation_size=2000
python scripts/data_process/sum.py             --max_length=4096 --validation_size=1000

# 2. Train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/h100_config.yaml \
    --main_process_port 25678 \
    granite_kvlink5_trainer.py

# 3. Evaluate on all four benchmarks
CKPT=training_res/kvlink_5_granite_8B/checkpoint-6000
for pos in 0 4 9; do
    python scripts/granite/nq.py --ckpt_path $CKPT --pos $pos --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
done
python scripts/granite/2wiki.py   --ckpt_path $CKPT --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
python scripts/granite/hqa.py     --ckpt_path $CKPT --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
python scripts/granite/musique.py --ckpt_path $CKPT --batch_size 4 --attn_type blocked --reencode_num 5 --hf True
```
