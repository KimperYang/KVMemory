"""
Loading dataset in a streaming way and allow resume from checkpoints
Reference: torchtitan/datasets/hf_dataset.py
"""
import os
from typing import List, Tuple

import datasets
import torch
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.data.input_preprocessor import bias_attention_preprocessor, custom_collate_bias
from src.torchtitan.datasets.tokenizer import Tokenizer
from src.torchtitan.logging import logger
from src.training.custom_trainer import CustomTrainerBiasAttn


def load_data_and_process_fn(
    data_component_name: str,
    preprocessor: bias_attention_preprocessor,
    training: bool = True,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        elif data_component_name == "text_mem":
            preprocessor_fn = preprocessor.process_textmem
        elif data_component_name == "text_inst":
            preprocessor_fn = preprocessor.process_textinst
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        if data_component_name in ["text_mem", "text_inst"]:
            remove_columns.append("num_tokens")
    elif data_component_name in ["sft", "sft_mem"]:
        data_path = f"dataset_cache/processed/daringanteater/{data_component_name}"
        if data_component_name == "sft":
            preprocessor_fn = preprocessor.process_sft
        elif data_component_name == "sft_mem":
            preprocessor_fn = preprocessor.process_sftmem
        else:
            raise NotImplementedError()
        remove_columns=["system", "mask", "dataset", "conversations"]
    elif data_component_name in ["qa", "qa_mem"]:
        data_path = f"dataset_cache/processed/2wiki/{data_component_name}"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_mem":
            preprocessor_fn = preprocessor.process_qamem
        else:
            raise NotImplementedError()
        remove_columns=["question", "context", "answer"]
    else:
        raise NotImplementedError()

    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    if training:
        ds = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    else:
        ds = data_component["test"]
    return ds, preprocessor_fn, remove_columns

class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Tokenizer,
        seq_len: int = 4096,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        preprocessor = bias_attention_preprocessor(tokenizer=tokenizer, max_len=seq_len)
        ds, preprocess_fn, columns_to_remove = load_data_and_process_fn(
            data_component_name=dataset_name,
            preprocessor=preprocessor,
            training=True,
        )

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.preprocess_fn = preprocess_fn
        self.columns_to_remove = columns_to_remove

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}






