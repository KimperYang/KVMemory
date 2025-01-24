from typing import Dict

import datasets
import torch
from absl.testing import parameterized
from transformers import AutoTokenizer

from src.data.attention import (
    construct_biased_attention_matrix,
    construct_biased_attention_matrix_v2,
)
from src.data.input_preprocessor import custom_collate_bias, sum_attention_preprocessor
from src.data.titan_preprocessor import (
    BlockAttnCollator,
    SumAttentionPreprocessor,
    make_segment_mask,
)
from src.data.titan_tokenizer import LLaMA32Tokenizer

LLAMA_tokenizer_path="data/titan_tokenizer/original/tokenizer.model"
DATASET_NAME_TO_PATH = {
    "pretrain_process": "dataset_cache/processed/fineweb/text",
    "text_inst": "dataset_cache/processed/fineweb/text_inst",
    "text_mem": "dataset_cache/processed/fineweb/text_mem",
    "tulu_process": "dataset_cache/processed/tulu/sft",
    "qa_process": "dataset_cache/processed/block_qa/qa",
    "qa_mem_process": "dataset_cache/processed/block_qa/qa_mem",
    "sft_mem_process": "dataset_cache/processed/daringanteater/sft_mem",
    "xsum_process": "dataset_cache/processed/xsum/xsum"
}
DATASET_NAME_TO_PROCESSOR = {
    "pretrain_process": "process_text",
    "text_inst": "process_textinst",
    "text_mem": "process_textmem",
    "tulu_process": "process_tulu",
    "qa_process": "process_qa",
    "qa_mem_process": "process_qamem",
    "sft_mem_process": "process_sftmem",
    "xsum_process": "process_xsum",
}

class SimpleTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.new_collator = BlockAttnCollator(pad_token_idx=0)

    @parameterized.parameters(
        (
            [0,1,2,3,4,5,6,7,8,9],
            [0,0,1,1,1,0,2,2,0,-1],
            9,
            [[2,5], [6,8]]
        ),
    )
    def test_masks_constructed_from_segment_ids(self, input_ids, segment_ids, input_length, biased_ranges):
        device = torch.device("cpu")
        ref_attn_mask = construct_biased_attention_matrix_v2(
            input_length,
            biased_ranges,
            10,
            device=device,
        )

        segment_ids = torch.LongTensor(segment_ids).view(1, -1)
        new_attn_mask = make_segment_mask(
            source_segments=segment_ids,
            target_segments=segment_ids,
            dtype=torch.bfloat16,
        )
        self.assertTrue(
            torch.allclose(new_attn_mask, ref_attn_mask, atol=1e-6),
            "Mismatch found in attention matrices between new and reference."
        )



class PreprocessorTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama_tokenizer = LLaMA32Tokenizer(LLAMA_tokenizer_path)
        cls.hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        special_token_configs = dict(
            special_token_start=128011,
            mem_start=128254,
            mem_end=128255,
            reencode_num=10,
        )
        cls.new_preprocessor = SumAttentionPreprocessor(
            cls.llama_tokenizer, max_len=4096,
            **special_token_configs,
        )
        cls.ref_preprocessor = sum_attention_preprocessor(
            cls.hf_tokenizer, max_len=4096,
            **special_token_configs,
        )

        cls.new_collator = BlockAttnCollator(pad_token_idx=0)

    def compare_processed_datasets(self, data_path, new_precess_fn, ref_process_fn):
        full_ds = datasets.load_from_disk(data_path)["train"]
        full_ds.cleanup_cache_files()
        ds = full_ds.select(range(10))
        del full_ds

        import random
        random.seed(222)
        new_ds = ds.map(new_precess_fn, batched=False, load_from_cache_file=False)
        random.seed(222)
        ref_ds = ds.map(ref_process_fn, batched=False, load_from_cache_file=False)

        keys_to_compare = ["input_ids", "labels"]
        for idx in range(len(new_ds)):
            new_example = new_ds[idx]
            ref_example = ref_ds[idx]
            for key in keys_to_compare:
                self.assertEqual(
                    new_example[key], ref_example[key],
                    f"Mismatch found in preprocessed examples at index {idx}"
                    f"and key{key}."
                )

        ref_processed_batch: Dict[str, torch.Tensor] = custom_collate_bias(batch=[example for example in ref_ds])
        new_processed_batch: Dict[str, torch.Tensor] = self.new_collator([example for example in new_ds])
        self.assertTrue(
            torch.equal(new_processed_batch["input_ids"], ref_processed_batch["input_ids"]),
            "Mismatch found in 'input_ids' between new and reference processed batches."
        )
        self.assertTrue(
            torch.equal(new_processed_batch["labels"], ref_processed_batch["labels"]),
            "Mismatch found in 'labels' between new and reference processed batches."
        )

        ref_attention_matrices = []
        max_length = max(ref_processed_batch['input_length'])
        for idx in range(len(ref_processed_batch['input_ids'])):
            mem_num = ref_processed_batch['mem_num'][idx]
            if mem_num == 0:
                biased_ranges = None
            else:
                biased_ranges = ref_processed_batch['biased_index'][idx][:mem_num]
            ref_attention_matrices.append(
                construct_biased_attention_matrix_v2(
                # construct_biased_attention_matrix(
                    ref_processed_batch['input_length'][idx],
                    biased_ranges,
                    max_length,
                    ref_processed_batch['input_ids'].device
                )
            )
        ref_attention_matrices = torch.stack(ref_attention_matrices)

        new_attention_matrices = make_segment_mask(
            source_segments=new_processed_batch["segment_ids"],
            target_segments=new_processed_batch["segment_ids"],
            dtype=torch.bfloat16,
        )
        # import ipdb
        # ipdb.set_trace()
        self.assertTrue(
            torch.allclose(new_attention_matrices, ref_attention_matrices, atol=1e-6),
            "Mismatch found in attention matrices between new and reference."
        )


    @parameterized.named_parameters(
        # ("pretrain_process", "pretrain_process"),
        # ("text_inst", "text_inst"),
        # ("text_mem", "text_mem"),
        # ("tulu_process", "tulu_process"),
        # ("sft_mem", "sft_mem_process"),
        # ("qa_process", "qa_process"),
        # ("qa_mem_process", "qa_mem_process"),
        ("xsum_process", "xsum_process"),
    )
    def test_processing_pipeline(self, dataset_name):
        # Retrieve data path and processor function dynamically
        data_path = DATASET_NAME_TO_PATH[dataset_name]
        process_fn_name = DATASET_NAME_TO_PROCESSOR[dataset_name]

        # Get preprocessing functions
        llama_fn = getattr(self.new_preprocessor, process_fn_name)
        hf_fn = getattr(self.ref_preprocessor, process_fn_name)

        # Compare processed datasets
        self.compare_processed_datasets(data_path, llama_fn, hf_fn)


