import datasets
import numpy as np
from absl.testing import absltest
from transformers import AutoTokenizer

from src.data.input_preprocessor import bias_attention_preprocessor as HF_PROCESSOR
from src.data.titan_preprocessor import bias_attention_preprocessor as LLAMA_PROCESSOR
from src.data.titan_tokenizer import LLaMA32Tokenizer

LLAMA_tokenizer_path="data/titan_tokenizer/original/tokenizer.model"

class InputTokenizationTest(absltest.TestCase):
    llama_tokenizer = LLaMA32Tokenizer(LLAMA_tokenizer_path)
    hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    llama_preprocessor = LLAMA_PROCESSOR(llama_tokenizer, max_len=4096)
    hf_preprocessor = HF_PROCESSOR(hf_tokenizer, max_len=4096)

    def test_tokenization(self,):
        data_path = "dataset_cache/processed/fineweb/text"
        ds = datasets.load_from_disk(data_path)
        test_examples = ds["text"][:10]
        for example in test_examples:
            llama_tokenized = self.llama_tokenizer(example, add_special_tokens=False)
            hf_tokenized = self.hf_tokenizer(example, add_special_tokens=False)
        self.assertEqual(
            llama_tokenized["input_ids"], hf_tokenized["input_ids"],
            f"Tokenized IDs differ for input: {example}"
        )

    def test_pretrain_process(self,):
        data_path = "dataset_cache/processed/fineweb/text"
        full_ds = datasets.load_from_disk(data_path)
        ds = full_ds.select(np.arange(10))
        del full_ds

        llama_preprocess_fn = self.llama_preprocessor.process_text
        hf_preprocess_fn = self.hf_preprocessor.process_text

        llama_ds = ds.map(llama_preprocess_fn, batched=False)
        hf_ds = ds.map(hf_preprocess_fn, batched=False,)

        for idx in range(len(llama_ds)):
            llama_example = llama_ds[idx]
            hf_example = hf_ds[idx]
            self.assertDictEqual(
                llama_example, hf_example,
                f"Mismatch found in preprocessed examples at index {idx}"
            )

    def test_tulu_process(self,):
        data_path = "dataset_cache/processed/tulu/sft"
        full_ds = datasets.load_from_disk(data_path)
        ds = full_ds.select(np.arange(10))
        del full_ds

        llama_preprocess_fn = self.llama_preprocessor.process_tulu
        hf_preprocess_fn = self.hf_preprocessor.process_tulu

        llama_ds = ds.map(llama_preprocess_fn, batched=False)
        hf_ds = ds.map(hf_preprocess_fn, batched=False,)

        for idx in range(len(llama_ds)):
            llama_example = llama_ds[idx]
            hf_example = hf_ds[idx]
            self.assertDictEqual(
                llama_example, hf_example,
                f"Mismatch found in preprocessed examples at index {idx}"
            )

    def test_qa_process(self,):
        data_path = "dataset_cache/processed/block_qa/qa"
        full_ds = datasets.load_from_disk(data_path)
        ds = full_ds.select(np.arange(10))
        del full_ds

        llama_preprocess_fn = self.llama_preprocessor.process_qa
        hf_preprocess_fn = self.hf_preprocessor.process_qa

        llama_ds = ds.map(llama_preprocess_fn, batched=False)
        hf_ds = ds.map(hf_preprocess_fn, batched=False,)

        for idx in range(len(llama_ds)):
            llama_example = llama_ds[idx]
            hf_example = hf_ds[idx]
            self.assertDictEqual(
                llama_example, hf_example,
                f"Mismatch found in preprocessed examples at index {idx}"
            )


    def test_qa_mem_process(self,):
        data_path = "dataset_cache/processed/block_qa/qa_mem"
        full_ds = datasets.load_from_disk(data_path)
        ds = full_ds.select(np.arange(10))
        del full_ds

        llama_preprocess_fn = self.llama_preprocessor.process_qamem
        hf_preprocess_fn = self.hf_preprocessor.process_qamem

        llama_ds = ds.map(llama_preprocess_fn, batched=False)
        hf_ds = ds.map(hf_preprocess_fn, batched=False,)

        for idx in range(len(llama_ds)):
            llama_example = llama_ds[idx]
            hf_example = hf_ds[idx]
            self.assertDictEqual(
                llama_example, hf_example,
                f"Mismatch found in preprocessed examples at index {idx}"
            )



