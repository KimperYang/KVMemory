import random
from typing import Dict

import torch
from transformers import PreTrainedTokenizerBase
from src.data.compress import insert_mem_tokens, get_position_id

class bias_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        current_position = sys_len
        all_input_ids = sys_tokens
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']
                memory_tokens = [128256] + memory_tokens + [128257]
                all_input_ids = all_input_ids + memory_tokens

                mem_len = len(memory_tokens)

                biased_index.append([current_position, current_position + mem_len])

                current_position += mem_len

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        last_q_ids = [128258] + self.tokenizer(last_q, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids


        seq_len = len(all_input_ids)
        ans_len = len(last_a_ids)
        labels = [-100] * (seq_len - ans_len) + last_a_ids

        if len(all_input_ids)>4096:
            print(f"sftmem Exceed: {len(all_input_ids)}")

        return {
            'input_ids': all_input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1: 
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_tokens = [128258] + user_tokens
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - 2 * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [128256] + split_memory_ids[i] + [128257]
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id)])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + 2 *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("concat_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('textinst')

        if len(concat_ids)>4096:
            print(f"textinst Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        # text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user_tokens = [128258]
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - 2 * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [128256] + split_memory_ids[i] + [128257]
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id)])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + 2 *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("remain_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('false')

        if len(concat_ids)>4096:
            print(f"textmem Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        current_index = sys_len
        biased_index = []

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer("<MEM_START>" + f"Document [{j+1}](Title: {title}) {text}\n<MEM_END>", add_special_tokens=False).input_ids

            biased_index.append([current_index, current_index + len(tem_id)])
            current_index += len(tem_id)

            input_ids += tem_id

        user = "<MEM_SUM><|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qamem Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': None
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
        }

def custom_collate_bias(batch):
    input_ids = []
    labels = []
    biased_index = []
    mem_num = []
    input_length = []
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
        input_ids.append(item['input_ids'] + [0] * (max_length - seq_length))
        labels.append(item['labels'] + [-100] * (max_length - seq_length))

        if item['biased_index'] is None:
            curr_biased_index =  [[0,0]] * (max_mem_num)
            biased_index.append(curr_biased_index)
        else:
            biased_index.append(item['biased_index'] + [[0,0]] * (max_mem_num - _mem_num))

    return {
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(labels),
        'biased_index': torch.LongTensor(biased_index),
        "input_length": torch.LongTensor(input_length),
        'mem_num': torch.LongTensor(mem_num),
    }

class reencode_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        reencode_num: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.reencode_num = reencode_num

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        current_position = sys_len
        all_input_ids = sys_tokens
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']
                memory_tokens = [128256] + memory_tokens + [128257] + [128258] * self.reencode_num
                all_input_ids = all_input_ids + memory_tokens

                mem_len = len(memory_tokens)

                biased_index.append([current_position, current_position + mem_len - self.reencode_num])

                current_position += mem_len

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids


        seq_len = len(all_input_ids)
        ans_len = len(last_a_ids)
        labels = [-100] * (seq_len - ans_len) + last_a_ids

        return {
            'input_ids': all_input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - (2 + self.reencode_num) * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [128256] + split_memory_ids[i] + [128257] + [128258] * self.reencode_num
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id) - self.reencode_num])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + (2 + self.reencode_num) *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("concat_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('textinst')

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        # text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        current_index = sys_len
        biased_index = []

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            tem_id = [128256] + tem_id + [128257] + [128258] * self.reencode_num

            biased_index.append([current_index, current_index + len(tem_id) - self.reencode_num])

            current_index += len(tem_id)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qamem Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': None
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        # user_tokens = [128258]
        user_tokens = []
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - (2 + self.reencode_num) * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [128256] + split_memory_ids[i] + [128257] + [128258] * self.reencode_num
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id) - self.reencode_num])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + (2 + self.reencode_num) *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("remain_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('false')

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
        }

class baseline_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        all_input_ids = sys_tokens

        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']
                all_input_ids = all_input_ids + memory_tokens

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids


        seq_len = len(all_input_ids)
        ans_len = len(last_a_ids)
        labels = [-100] * (seq_len - ans_len) + last_a_ids

        return {
            'input_ids': all_input_ids,
            'labels': labels
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1: 
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_tokens = user_tokens
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens + memory_ids + user_tokens + remaining_ids

        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        return {
            'input_ids': concat_ids,
            'labels': labels
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        # text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - sys_len]

        mem_len = random.randint(500, 1500)

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens + memory_ids + remaining_ids

        labels = [-100] * (sys_len + mem_len) + remaining_ids

        return {
            'input_ids': concat_ids,
            'labels': labels
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id
        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please summarize the text based on the information given.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        system_input_ids = system_input_ids

        document_id = self.tokenizer(example['document'], add_special_tokens=False).input_ids

        user =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids

        ans_id = self.tokenizer(example['summary'] + "<|eot_id|>", add_special_tokens=False).input_ids

        input_ids = system_input_ids + document_id + user_id + ans_id
        labels = [-100] * (len(input_ids) - len(ans_id)) + ans_id

        if len(input_ids)>4096:
            print(f"xsum Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels
        }

def custom_collate_baseline(batch):
    input_ids = []
    labels = []
    input_length = []

    for item in batch:
        input_length.append(len(item['input_ids']))

    max_length = max(input_length)

    for item in batch:
        seq_length = len(item['input_ids'])
        input_ids.append(item['input_ids'] + [0] * (max_length - seq_length))
        labels.append(item['labels'] + [-100] * (max_length - seq_length))

    return {
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(labels)
    }

class seq_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        special_token_start: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_token_start = special_token_start

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        current_position = sys_len
        all_input_ids = sys_tokens
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']
                memory_tokens = [self.special_token_start + idx] + memory_tokens + [self.special_token_start + idx + 1]
                all_input_ids = all_input_ids + memory_tokens

                mem_len = len(memory_tokens)

                biased_index.append([current_position + 1, current_position + mem_len - 1])

                current_position += mem_len

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids


        seq_len = len(all_input_ids)
        ans_len = len(last_a_ids)
        labels = [-100] * (seq_len - ans_len) + last_a_ids

        if len(all_input_ids)>4096:
            print(f"sftmem Exceed: {len(all_input_ids)}")

        return {
            'input_ids': all_input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1: 
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - 2 * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [self.special_token_start + 2 * i] + split_memory_ids[i] + [self.special_token_start + 2 * i + 1]
            concat_ids += tem_mem_id

            biased_index.append([bias_position + 1, bias_position + len(tem_mem_id) - 1])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + 2 *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("concat_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('textinst')

        if len(concat_ids)>4096:
            print(f"textinst Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        # text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user_tokens = []
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - 2 * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [self.special_token_start + 2 * i] + split_memory_ids[i] + [self.special_token_start + 2 * i + 1]
            concat_ids += tem_mem_id

            biased_index.append([bias_position + 1, bias_position + len(tem_mem_id) - 1])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + 2 *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if len(concat_ids)>4096:
            print(f"textmem Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        current_index = sys_len
        biased_index = []

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            tem_id = [self.special_token_start + 2 * j] + tem_id + [self.special_token_start + 2 * j + 1]

            biased_index.append([current_index + 1, current_index + len(tem_id) - 1])
            current_index += len(tem_id)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qamem Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': None
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
        }
    
class block_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        do_shuffle: bool
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_shuffle = do_shuffle

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        current_position = sys_len
        all_input_ids = sys_tokens
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']
                all_input_ids = all_input_ids + memory_tokens

                mem_len = len(memory_tokens)

                biased_index.append([current_position, current_position + mem_len])

                current_position += mem_len

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids


        seq_len = len(all_input_ids)
        ans_len = len(last_a_ids)
        labels = [-100] * (seq_len - ans_len) + last_a_ids

        if len(all_input_ids)>4096:
            print(f"sftmem Exceed: {len(all_input_ids)}")

        return {
            'input_ids': all_input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1: 
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i]
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id)])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if len(concat_ids)>4096:
            print(f"textinst Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        # text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user_tokens = []
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i]
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id)])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if len(concat_ids)>4096:
            print(f"textmem Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        current_index = sys_len
        biased_index = []

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            biased_index.append([current_index, current_index + len(tem_id)])
            current_index += len(tem_id)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qamem Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': None
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please summarize the text based on the information given.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        document_id = self.tokenizer(example['document'], add_special_tokens=False).input_ids
        chunks = [document_id[i:i+100] for i in range(0, len(document_id), 100)]

        current_index = sys_len
        biased_index = []

        for j in range(len(chunks)):
            
            tem_id = chunks[j]

            biased_index.append([current_index, current_index + len(tem_id)])

            current_index += len(tem_id)

            input_ids += tem_id

        user =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['summary'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        labels = [-100] * (len(input_ids) - len(ans_id)) + ans_id

        if len(input_ids)>4096:
            print(f"xsum Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

class sum_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        special_token_start: int,
        mem_start: int,
        mem_end: int,
        reencode_num: int,
        do_shuffle: bool
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_token_start = special_token_start
        self.mem_start = mem_start
        self.mem_end = mem_end
        self.reencode_num = reencode_num
        self.do_shuffle = do_shuffle

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_tokens += [self.mem_start]
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        current_position = sys_len
        all_input_ids = sys_tokens
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']

                for sub_idx in range(self.reencode_num):
                    memory_tokens = memory_tokens + [self.special_token_start + int(idx / 2) * self.reencode_num + sub_idx]

                all_input_ids = all_input_ids + memory_tokens

                mem_len = len(memory_tokens)

                biased_index.append([current_position, current_position + mem_len - self.reencode_num])

                current_position += mem_len

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        last_q_ids = self.tokenizer(last_q, add_special_tokens= False)['input_ids']
        last_q_ids = [self.mem_end] + last_q_ids

        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids


        seq_len = len(all_input_ids)
        ans_len = len(last_a_ids)
        labels = [-100] * (seq_len - ans_len) + last_a_ids

        if len(all_input_ids)>4096:
            print(f"sftmem Exceed: {len(all_input_ids)}")

        return {
            'input_ids': all_input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1: 
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_tokens = sys_tokens + [self.mem_start]
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_tokens = [self.mem_end] + user_tokens
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - self.reencode_num * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i]
            for sub_idx in range(self.reencode_num):
                tem_mem_id = tem_mem_id + [self.special_token_start + self.reencode_num * i + sub_idx]
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id) - self.reencode_num])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + self.reencode_num *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("concat_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('textinst')

        if len(concat_ids)>4096:
            print(f"textinst Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        # text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_tokens = sys_tokens + [self.mem_start]
        sys_len = len(sys_tokens)

        user_tokens = [self.mem_end]
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - self.reencode_num * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i]
            for sub_idx in range(self.reencode_num):
                tem_mem_id = tem_mem_id + [self.special_token_start + self.reencode_num * i + sub_idx]

            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id) - self.reencode_num])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + self.reencode_num *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if len(concat_ids)>4096:
            print(f"textmem Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        system_input_ids = system_input_ids + [self.mem_start]
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        current_index = sys_len
        biased_index = []

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            for sub_idx in range(self.reencode_num):
                tem_id = tem_id + [self.special_token_start + self.reencode_num * j + sub_idx]

            biased_index.append([current_index, current_index + len(tem_id) - self.reencode_num])
            current_index += len(tem_id)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        user_id = [self.mem_end] + user_id
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qamem Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': None
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please summarize the text based on the information given.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        system_input_ids = system_input_ids + [self.mem_start]
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        document_id = self.tokenizer(example['document'], add_special_tokens=False).input_ids
        chunks = [document_id[i:i+100] for i in range(0, len(document_id), 100)]

        current_index = sys_len
        biased_index = []

        for j in range(len(chunks)):
            
            tem_id = chunks[j]

            for sub_idx in range(self.reencode_num):
                tem_id = tem_id + [self.special_token_start + self.reencode_num * j + sub_idx]

            biased_index.append([current_index, current_index + len(tem_id) - self.reencode_num])

            current_index += len(tem_id)

            input_ids += tem_id

        user =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        user_id = [self.mem_end] + user_id
        input_ids += user_id

        ans_id = self.tokenizer(example['summary'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        labels = [-100] * (len(input_ids) - len(ans_id)) + ans_id

        if len(input_ids)>4096:
            print(f"xsum Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

class compress_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        compress_tokens: list[int]
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.compress_len = len(compress_tokens)

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You're an assistant who answer the question with the knowledge provided "
            "in the prompt<|eot_id|>"
        )
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]

        current_position = sys_len
        all_input_ids = sys_tokens
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = (
                    "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"]
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    + conversation[idx + 1]["value"] + "<|eot_id|>"
                )
                memory_tokens = self.tokenizer(text, add_special_tokens = False)['input_ids']

                all_input_ids = all_input_ids + memory_tokens

                mem_len = len(memory_tokens)

                biased_index.append([current_position, current_position + mem_len])

                current_position += mem_len

        last_q = (
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            conversation[len(conversation) - 2]["value"] +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        last_q_ids = self.tokenizer(last_q, add_special_tokens= False)['input_ids']

        all_input_ids = all_input_ids + last_q_ids

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False)['input_ids']
        all_input_ids = all_input_ids + last_a_ids

        ans_len = len(last_a_ids)

        new_input_len = len(all_input_ids) + 2 + len(biased_index) * self.compress_len
        labels = [-100] * (new_input_len - ans_len) + last_a_ids

        if len(all_input_ids)>4096:
            print(f"sftmem Exceed: {len(all_input_ids)}")

        return {
            'input_ids': all_input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= self.max_len: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > self.max_len - 1: 
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        # attention_matrix = construct_biased_attention_matrix(len(input_ids_list), [])
        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,20)

        input_ids = input_ids[:self.max_len - user_len - sys_len - (2 + mem_num * self.compress_len)]

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]


        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        # print(len(remaining_ids), len(input_ids), mem_len)

        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i]
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id)])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids

        new_input_len = len(concat_ids) + 2 + mem_num * self.compress_len
        labels = [-100] * (new_input_len - len(remaining_ids)) + remaining_ids

        if len(concat_ids)>4096:
            print(f"textinst Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):

        sys = "<|begin_of_text|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user_tokens = []
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        mem_len = random.randint(500, 1500)
        mem_num = random.randint(5,40)

        input_ids = input_ids[:self.max_len - user_len - sys_len - (2 + mem_num * self.compress_len)]

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i]

            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id)])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids

        new_input_len = len(concat_ids) + 2 + mem_num * self.compress_len
        labels = [-100] * (new_input_len - len(remaining_ids)) + remaining_ids

        if len(concat_ids)>4096:
            print(f"textmem Exceed: {len(concat_ids)}")

        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        system_input_ids = system_input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        current_index = sys_len
        biased_index = []

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            biased_index.append([current_index, current_index + len(tem_id)])
            current_index += len(tem_id)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        new_input_len = len(input_ids) + 2 + len(biased_index) * self.compress_len
        labels = [-100] * (new_input_len - len(ans_id)) + ans_id

        if len(input_ids)>4096:
            print(f"qamem Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        if len(input_ids)>4096:
            print(f"qa Exceed: {len(input_ids)}")


        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': None
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example['messages']
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        input_ids_list = system_input_ids
        labels = [-100] * sys_len
        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["content"]  + "<|eot_id|>"

                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["content"]
                tokenized = self.tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > self.max_len - 1:
                    input_ids = input_ids[:self.max_len - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(input_ids_list)>4096:
            print(f"sft Exceed: {len(input_ids_list)}")

        return {
            'input_ids': input_ids_list,
            'labels': labels,
            'biased_index': None
        }
    
    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please summarize the text based on the information given.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        system_input_ids = system_input_ids
        input_ids = system_input_ids
        sys_len = len(system_input_ids)

        document_id = self.tokenizer(example['document'], add_special_tokens=False).input_ids
        chunks = [document_id[i:i+100] for i in range(0, len(document_id), 100)]

        current_index = sys_len
        biased_index = []

        for j in range(len(chunks)):

            tem_id = chunks[j]

            biased_index.append([current_index, current_index + len(tem_id)])
            current_index += len(tem_id)

            input_ids += tem_id

        user =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        ans_id = self.tokenizer(example['summary'] + "<|eot_id|>", add_special_tokens=False).input_ids
        input_ids += ans_id

        new_input_len = len(input_ids) + 2 + len(biased_index) * self.compress_len
        labels = [-100] * (new_input_len - len(ans_id)) + ans_id

        if len(input_ids)>4096:
            print(f"xsum Exceed: {len(input_ids)}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'biased_index': biased_index
        }

def custom_collate_compress(batch, compress_tokens):

    input_ids = []
    labels = []
    biased_index = []
    mem_num = []
    position_ids = []
    input_length = []
    for item in batch:
        if item['biased_index'] is not None:
            mem_num.append(len(item['biased_index']))
        else:
            mem_num.append(0)
        input_length.append(len(item['labels']))

    max_mem_num = max(mem_num)
    max_length = max(input_length)

    for item in batch:

        if item['biased_index'] is not None:
            # shift_input_ids, shift_biased_index = insert_mem_tokens(item['input_ids'], item['biased_index'], list(range(128011, 128031)), 128254, 128255)
            shift_input_ids, shift_biased_index = insert_mem_tokens(item['input_ids'], item['biased_index'], compress_tokens, 128254, 128255)
        else:
            shift_input_ids = item['input_ids']
            shift_biased_index = []

        if len(shift_input_ids) != len(item['labels']):
            print("There is some problem shifting the input")

        seq_length = len(shift_input_ids)

        _mem_num = len(shift_biased_index)

        input_ids.append(shift_input_ids + [0] * (max_length - seq_length))
        labels.append(item['labels'] + [-100] * (max_length - seq_length))

        _position_id = get_position_id(shift_input_ids, shift_biased_index)
        position_ids.append(_position_id + [0] * (max_length - seq_length))

        biased_index.append(shift_biased_index + [[0,0]] * (max_mem_num - _mem_num))

    return {
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(labels),
        'biased_index': torch.LongTensor(biased_index),
        'input_length': torch.LongTensor(input_length),
        'position_ids': torch.LongTensor(position_ids),
        'mem_num': torch.LongTensor(mem_num),
    }