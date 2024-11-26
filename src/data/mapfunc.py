import random
import torch
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
from src.data.attention import construct_biased_attention_matrix
class multi_kv_preprocessor():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        """
        Apply chat template to the conversation between a user and the assistant.
        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
            max_len: the maximum length of the processed text.
            eot_id: the token id of the <turn_end> token.
            prepend_eos: whether prepend an eos token to the processed text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'sftmem'
        conversation = example["conversations"]
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        
        memory_ids = []
        memory_positions = []
        current_position = sys_len
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
                memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
                memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
                memory_ids.append(memory_tokens[0])

                mem_len = memory_tokens.size(1)
                memory_positions.append(torch.arange(current_position, current_position + mem_len))
                current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
                'input_ids': remaining_ids,
                'labels': labels,
                'dataset_id': dataset_id,
                'memory_position': memory_positions,
                'split_memory_id': memory_ids,
                'sys_id': sys_tokens
            }    

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids

        input_ids_list = system_input_ids
        labels = [-100] * len(system_input_ids)

        for i in range(len(conversation)):
            
            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 
                    # t = " </s><s>[INST] " + conversation[i]["value"]  + " [/INST] " 
                
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

                # # add </s>
                # input_ids = torch.cat([input_ids, self.eos_token.to(input_ids.device)], dim = 1)
                # attention_msk = torch.cat([attention_msk, self.mask_token.to(attention_msk.device)], dim = 1) 
                input_ids += [128009]

                # if len(mask) + input_ids.size(1) > self.max_len: 
                #     input_ids = input_ids[:, :self.max_len - len(mask)]
                #     attention_msk = attention_msk[:, :self.max_len - len(mask)]


                labels.extend(input_ids)
                
                # print(input_ids.size(1),attention_msk.size(1), input_ids.device, attention_msk.device)

                input_ids_list += input_ids
        
        # input_ids = torch.cat(input_ids_list, dim=1)
        # attention_mask = torch.cat(attention_mask_list, dim=1)

        # tensor_input_ids = torch.tensor([input_ids_list])
        # tensor_attention_mask = torch.tensor([attention_mask_list])
        
        return {
            'input_ids': [input_ids_list],
            'labels': [labels],
            'dataset_id': 'sft',
            'memory_position': None,
            'split_memory_id': None,
            'sys_id': None
        }
    
    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False, return_tensors= "pt")['input_ids']
        user_tokens = torch.cat([torch.tensor([[128258]]), user_tokens], dim = 1)
        user_len = user_tokens.size(1)

        text = example["text"]
        tokenized = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")
        input_ids = tokenized.input_ids
        dataset_id = 'textinst'

        # allocate space for "<MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n". [128009, 128006, 78191, 128007, 271]
        input_ids = input_ids[:, :self.max_len - user_len - sys_len] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 2 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]

        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)).to(split_input_ids.device), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0)).to(split_input_ids.device)], dim=1)

        mem_len = mem_len + 2 * num_memory
        each_mem_len = each_mem_len + 2
        
        memory_position = torch.arange(sys_len, sys_len + mem_len).unsqueeze(0)
        # memory_positions = torch.cat([memory_position] * input_ids.size(0)
        memory_position_batch = memory_position.reshape(-1, each_mem_len)

        labels = torch.cat([torch.tensor([[-100] * user_len]), remaining_ids], dim=1)
        # add <MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        remaining_ids = torch.cat([user_tokens, remaining_ids], dim=1)
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'dataset_id': dataset_id,
            'memory_position': memory_position_batch,
            'split_memory_id': split_input_ids,
            'sys_id': sys_tokens
        }
     
    def process_text(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'text'
        text_tokens = self.tokenizer(example["text"], add_special_tokens= False, return_tensors= "pt")['input_ids']
        text_tokens = text_tokens[:, :self.max_len - 1]
        remaining_ids = torch.cat([torch.tensor([[128000]]).to(text_tokens.device), text_tokens], dim = 1)

        labels = torch.cat([torch.tensor([[-100]]).to(text_tokens.device), text_tokens], dim = 1)
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'dataset_id': dataset_id,
            'memory_position': None,
            'split_memory_id': None,
            'sys_id': None
        }
    
    def process_textmem(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'textmem'

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")["input_ids"]
    
        input_ids = input_ids[:, :self.max_len - 2] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 2 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]

        # add <|eot_id|>
        # remaining_ids = torch.cat([remaining_ids, torch.tensor([[128009]]).to(remaining_ids.device)], dim=1)

 
        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)).to(split_input_ids.device), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0)).to(split_input_ids.device)], dim=1)

        mem_len = mem_len + 2 * num_memory
        each_mem_len = each_mem_len + 2
        
        memory_position = torch.arange(1, 1 + mem_len).unsqueeze(0)
        # memory_positions = torch.cat([memory_position] * input_ids.size(0)
        memory_position_batch = memory_position.reshape(-1, each_mem_len)
        
        labels = torch.cat([torch.tensor([[-100]]), remaining_ids], dim = 1)
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'dataset_id': dataset_id,
            'memory_position': memory_position_batch,
            'split_memory_id': split_input_ids,
            'sys_id': torch.tensor([[128000]])
        }

    def process_raft_nqmem(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'nqmem'
        memory_text = example['context']['sentences'][0]

        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the documents retrieved below. Some documents may be irrelevant to the question.<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)
        
        memory_ids = []
        memory_positions = []
        current_position = sys_len

        for idx in range(len(memory_text)):
            text = memory_text[idx]
            memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + example['question'][3:] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = example['cot_answer'] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
                'input_ids': remaining_ids,
                'labels': labels,
                'dataset_id': dataset_id,
                'memory_position': memory_positions,
                'split_memory_id': memory_ids,
                'sys_id': sys_tokens
            }
    
    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'xsum'
        memory_text = example['document'].split('\n')

        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who helps to summarize the following passages.<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)
        
        memory_ids = []
        memory_positions = []
        current_position = sys_len

        for idx in range(len(memory_text)):
            text = memory_text[idx]
            memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\nSummarize the text provided above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = example['summary'] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
                'input_ids': remaining_ids,
                'labels': labels,
                'dataset_id': dataset_id,
                'memory_position': memory_positions,
                'split_memory_id': memory_ids,
                'sys_id': sys_tokens
            }
        
    # def process_daring_anteater(
    #     self,
    #     example: Dict[str, str],
    # ):
    #     """
    #     The entry point of the preprocessor for nvidia/Daring-Anteater dataset.
    #     """
    #     conversation = example["conversations"]
    #     texts = []
    #     roles = []
    #     if len(conversation) % 2 != 0:
    #         if conversation[0]["from"] == "Assistant":
    #             conversation = conversation[1:]
    #         elif conversation[-1]["from"] == "User":
    #             conversation = conversation[:-1]
    #         else:
    #             conversation = conversation[:-1]
    #     for idx in range(0, len(conversation), 2):
    #         if (
    #             conversation[idx]["from"] == "User" and
    #             conversation[idx + 1]["from"] == "Assistant"
    #         ):
    #             texts.append(conversation[idx]["value"])
    #             texts.append(conversation[idx + 1]["value"])
    #             roles += [0, 1]

    #     input_ids, labels = self.process_single_conversation(texts, roles)
    #     return {
    #         "input_ids": [input_ids],
    #         "labels": [labels],
    #         'dataset_id':'sft',
    #         'memory_position': None,
    #         'split_memory_id': None,
    #         'sys_id': None
    #     }

    # def process_single_conversation(
    #     self,
    #     texts: List[str],
    #     roles: List[str],
    # ) -> Any:
    #     """
    #     Apply chat template to the conversation between a user and the assistant.
    #     Args:
    #         texts: the message from user and assistant.
    #         roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
    #     Returns:
    #         input_ids: the input_ids with chat template applied
    #         labels: the labels for training. Simply set the positions of user messages to -100.
    #     """
    #     assert roles[0] == 0
    #     assert roles[1] == 1
    #     input_ids = []
    #     labels = []
    #     system_prompt = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"
    #     # system_prompt = system_prompt.replace("\n", "<n>")
    #     prefix_ids = self.tokenizer(
    #         system_prompt,
    #         padding=False,
    #         truncation=False,
    #         return_tensors=None,
    #     )["input_ids"]
    #     input_ids += prefix_ids
    #     labels += [-100] * len(prefix_ids)

    #     for idx in range(0, len(texts), 2):
    #         user_text = texts[idx]
    #         assistant_text = texts[idx + 1]
    #         assert roles[idx] == 0
    #         assert roles[idx + 1] == 1

    #         user_text = " user\n" + user_text 
    #         assistant_text = " assistant\n" + assistant_text

    #         # replace \n with <n>
    #         # user_text = user_text.replace("\n", "<n>")
    #         # assistant_text = assistant_text.replace("\n", "<n>")

    #         user_text_ids = self.tokenizer(
    #             user_text,
    #             add_special_tokens=False,
    #             padding=False,
    #             truncation=False,
    #             return_tensors=None,
    #         )["input_ids"]
    #         assistant_text_ids = self.tokenizer(
    #             assistant_text,
    #             add_special_tokens=False,
    #             padding=False,
    #             truncation=False,
    #             return_tensors=None,
    #         )["input_ids"]

    #         remaining_length = self.max_len - len(input_ids)
    #         if remaining_length <= 0:
    #             break
    #         if len(user_text_ids) + len(assistant_text_ids) <= remaining_length:
    #             input_ids = input_ids + user_text_ids + assistant_text_ids
    #             labels += [-100] * len(user_text_ids)
    #             labels += assistant_text_ids
    #         else:
    #             if len(input_ids) == 0:
    #                 if len(user_text_ids) >= remaining_length:
    #                     input_ids += user_text_ids
    #                     labels += [-100] * len(user_text_ids)
    #                     break
    #                 else:
    #                     input_ids += user_text_ids
    #                     labels += [-100] * len(user_text_ids)
    #                     input_ids += assistant_text_ids[:remaining_length - len(user_text_ids)]
    #                     labels += assistant_text_ids[:remaining_length - len(user_text_ids)]

    #     return input_ids, labels

class baseline_preprocessor():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        """
        Apply chat template to the conversation between a user and the assistant.
        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
            max_len: the maximum length of the processed text.
            eot_id: the token id of the <turn_end> token.
            prepend_eos: whether prepend an eos token to the processed text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'sftmem'
        conversation = example["conversations"]
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        
        prefix_num = sys_tokens.size(1)
        input_ids = sys_tokens

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
                memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']

                prefix_num += memory_tokens.size(1)
                input_ids = torch.cat([input_ids, memory_tokens], dim = 1)

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']

        input_ids = torch.cat([input_ids, last_q_ids], dim = 1)
        prefix_num += last_q_ids.size(1)
        labels = torch.tensor([[-100] * prefix_num])

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        input_ids = torch.cat([input_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
                'input_ids': input_ids,
                'labels': labels,
                'dataset_id': dataset_id,
                'memory_position': None,
                'split_memory_id': None,
                'sys_id': sys_tokens
            }    

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids

        input_ids_list = system_input_ids
        labels = [-100] * len(system_input_ids)

        for i in range(len(conversation)):
            
            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 
                    # t = " </s><s>[INST] " + conversation[i]["value"]  + " [/INST] " 
                
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
    
        
        return {
            'input_ids': [input_ids_list],
            'labels': [labels],
            'dataset_id': 'sft',
            'memory_position': None,
            'split_memory_id': None,
            'sys_id': None
        }
    
    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False, return_tensors= "pt")['input_ids']
        user_len = user_tokens.size(1)

        text = example["text"]
        tokenized = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")
        text_ids = tokenized.input_ids
        dataset_id = 'textinst'

        # allocate space for "<MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n". [128009, 128006, 78191, 128007, 271]
        text_ids = text_ids[:, :self.max_len - user_len - sys_len] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        memory_ids = text_ids[:, :mem_len]
        remaining_ids = text_ids[:, mem_len:]

        labels = torch.cat([torch.tensor([[-100] * (sys_len + mem_len + user_len)]), remaining_ids], dim=1)
        input_ids = torch.cat([sys_tokens, memory_ids, user_tokens, remaining_ids], dim=1)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'dataset_id': dataset_id,
            'memory_position': None,
            'split_memory_id': None,
            'sys_id': sys_tokens
        }
     
    def process_text(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'text'
        text_tokens = self.tokenizer(example["text"], add_special_tokens= False, return_tensors= "pt")['input_ids']
        text_tokens = text_tokens[:, :self.max_len - 1]
        remaining_ids = torch.cat([torch.tensor([[128000]]).to(text_tokens.device), text_tokens], dim = 1)

        labels = torch.cat([torch.tensor([[-100]]).to(text_tokens.device), text_tokens], dim = 1)
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'dataset_id': dataset_id,
            'memory_position': None,
            'split_memory_id': None,
            'sys_id': None
        }
    
    def process_textmem(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'textmem'

        text = example["text"]
        text_ids = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")["input_ids"]
    
        text_ids = text_ids[:, :self.max_len - 1] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len
        
        memory_ids = text_ids[:, :mem_len]
        remaining_ids = text_ids[:, mem_len:]

        labels = torch.cat([torch.tensor([[-100] * (1 + mem_len)]), remaining_ids], dim = 1)
        input_ids = torch.cat([torch.tensor([[128000]]), memory_ids, remaining_ids], dim = 1)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'dataset_id': dataset_id,
            'memory_position': None,
            'split_memory_id': None,
            'sys_id': torch.tensor([[128000]])
        }

    def process_raft_nqmem(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'nqmem'
        memory_text = example['context']['sentences'][0]

        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the documents retrieved below. Some documents may be irrelevant to the question.<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)
        
        memory_ids = []
        memory_positions = []
        current_position = sys_len

        for idx in range(len(memory_text)):
            text = memory_text[idx]
            memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + example['question'][3:] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = example['cot_answer'] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
                'input_ids': remaining_ids,
                'labels': labels,
                'dataset_id': dataset_id,
                'memory_position': memory_positions,
                'split_memory_id': memory_ids,
                'sys_id': sys_tokens
            } 

class multi_kv_batch_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        """
        Apply chat template to the conversation between a user and the assistant.
        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
            max_len: the maximum length of the processed text.
            eot_id: the token id of the <turn_end> token.
            prepend_eos: whether prepend an eos token to the processed text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        
        memory_ids = [sys_tokens[0]]
        memory_positions = [torch.arange(0,sys_len)]
        current_position = sys_len
        max_mem_len = sys_len
        mem_nums = 1
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
                memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
                memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
                memory_ids.append(memory_tokens[0])

                mem_len = memory_tokens.size(1)
                memory_positions.append(torch.arange(current_position, current_position + mem_len))
                current_position += mem_len

                if mem_len > max_mem_len:
                    max_mem_len = mem_len

                mem_nums += 1

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
                'input_ids': remaining_ids,
                'labels': labels,
                'memory_length': max_mem_len,
                'memory_position': memory_positions,
                'split_memory_id': memory_ids,
                'memory_nums': mem_nums
            }    

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = self.tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids
        sys_len = len(system_input_ids)

        # input_ids_list = system_input_ids
        # labels = [-100] * len(system_input_ids)
        input_ids_list = []
        labels = []
        for i in range(len(conversation)):
            
            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 
                    # t = " </s><s>[INST] " + conversation[i]["value"]  + " [/INST] " 
                
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
        
        return {
            'input_ids': [input_ids_list],
            'labels': [labels],
            'memory_length': sys_len,
            'memory_position': [list(range(sys_len))],
            'split_memory_id': [system_input_ids],
            'memory_nums': 1
        }
    
    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False, return_tensors= "pt")['input_ids']
        user_tokens = torch.cat([torch.tensor([[128258]]), user_tokens], dim = 1)
        user_len = user_tokens.size(1)

        text = example["text"]
        tokenized = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")
        input_ids = tokenized.input_ids

        # allocate space for "<MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n". [128009, 128006, 78191, 128007, 271]
        input_ids = input_ids[:, :self.max_len - user_len - sys_len] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 2 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]

        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)).to(split_input_ids.device), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0)).to(split_input_ids.device)], dim=1)

        mem_len = mem_len + 2 * num_memory
        each_mem_len = each_mem_len + 2
        
        memory_position = torch.arange(sys_len, sys_len + mem_len).unsqueeze(0)
        # memory_positions = torch.cat([memory_position] * input_ids.size(0)
        memory_position = memory_position.reshape(-1, each_mem_len)

        memory_ids_list = split_input_ids.tolist()
        split_memory_ids_batch = sys_tokens.tolist() + memory_ids_list
        memory_position_list = memory_position.tolist()
        memory_position_batch = [list(range(sys_len))] + memory_position_list


        labels = torch.cat([torch.tensor([[-100] * user_len]), remaining_ids], dim=1)
        # add <MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        remaining_ids = torch.cat([user_tokens, remaining_ids], dim=1)
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'memory_length': max(each_mem_len, sys_len),
            'memory_position': memory_position_batch,
            'split_memory_id': split_memory_ids_batch,
            'memory_nums': num_memory + 1
        }
     
    def process_text(
        self,
        example: Dict[str, str],
    ):
        text_tokens = self.tokenizer(example["text"], add_special_tokens= False, return_tensors= "pt")['input_ids']
        text_tokens = text_tokens[:, :self.max_len - 1]
        split_memory_ids_batch = [[128000]]
        memory_position_batch = [[0]]
        remaining_ids =  text_tokens
        labels = text_tokens
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'memory_length': 1,
            'memory_position': memory_position_batch,
            'split_memory_id': split_memory_ids_batch,
            'memory_nums': 1
        }
    
    def process_textmem(
        self,
        example: Dict[str, str],
    ):
        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")["input_ids"]
    
        input_ids = input_ids[:, :self.max_len - 2] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 2 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]

        # add <|eot_id|>
        # remaining_ids = torch.cat([remaining_ids, torch.tensor([[128009]]).to(remaining_ids.device)], dim=1)

 
        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)).to(split_input_ids.device), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0)).to(split_input_ids.device)], dim=1)

        mem_len = mem_len + 2 * num_memory
        each_mem_len = each_mem_len + 2
        
        memory_position = torch.arange(1, 1 + mem_len).unsqueeze(0)
        # memory_positions = torch.cat([memory_position] * input_ids.size(0)
        memory_position = memory_position.reshape(-1, each_mem_len)
        
        memory_ids_list = split_input_ids.tolist()
        split_memory_ids_batch = [[128000]] + memory_ids_list
        memory_position_list = memory_position.tolist()
        memory_position_batch = [[0]] + memory_position_list
        
        labels = torch.cat([torch.tensor([[-100]]), remaining_ids], dim = 1)
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        
        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'memory_length': each_mem_len,
            'memory_position': memory_position_batch,
            'split_memory_id': split_memory_ids_batch,
            'memory_nums': 1 + num_memory
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'xsum'
        memory_text = example['document'].split('\n')

        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who helps to summarize the following passages.<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)
        
        memory_ids = [sys_tokens[0]]
        memory_positions = [torch.arange(0,sys_len)]
        current_position = sys_len

        max_memory_length = sys_len
        for idx in range(len(memory_text)):
            text = memory_text[idx]
            memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

            if mem_len > max_memory_length:
                max_memory_length = mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\nSummarize the text provided above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = example['summary'] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'memory_length': max_memory_length,
            'memory_position': memory_positions,
            'split_memory_id': memory_ids,
            'memory_nums': len(memory_text) + 1
            }
    
class bias_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        """
        Apply chat template to the conversation between a user and the assistant.
        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
            max_len: the maximum length of the processed text.
            eot_id: the token id of the <turn_end> token.
            prepend_eos: whether prepend an eos token to the processed text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        
        current_position = sys_len
        id_list = [sys_tokens]
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
                memory_tokens = self.tokenizer(text, add_special_tokens = False, return_tensors = "pt")['input_ids']
                memory_tokens = torch.cat([torch.tensor([[128256]]), memory_tokens, torch.tensor([[128257]])], dim = 1)
                id_list.append(memory_tokens)

                mem_len = memory_tokens.size(1)

                biased_index.append([current_position, current_position + mem_len])

                current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        last_q_ids = torch.cat([torch.tensor([[128258]]), last_q_ids], dim = 1)
        id_list.append(last_q_ids)

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        id_list.append(last_a_ids)    

        concat_ids = torch.cat(id_list, dim =1)

        seq_len = concat_ids.size(1)
        ans_len = last_a_ids.size(1)
        labels = torch.cat([torch.tensor([[-100] * (seq_len - ans_len)]), last_a_ids], dim = 1)

        # attention_matrix = construct_biased_attention_matrix(seq_len, biased_index)
        return {
                'input_ids': concat_ids,
                'labels': labels,
                'biased_index': biased_index
                # 'attention_matrix': attention_matrix
            }    

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
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
            'input_ids': [input_ids_list],
            'labels': [labels],
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }
    
# Todo
    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False, return_tensors= "pt")['input_ids']
        user_tokens = torch.cat([torch.tensor([[128258]]), user_tokens], dim = 1)
        user_len = user_tokens.size(1)

        text = example["text"]
        tokenized = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")
        input_ids = tokenized.input_ids

        # allocate space for "<MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n". [128009, 128006, 78191, 128007, 271]
        input_ids = input_ids[:, :self.max_len - user_len - sys_len] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 2 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]
        ans_len = remaining_ids.size(1)
        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0))], dim=1)

        mem_len = mem_len + 2 * num_memory
        each_mem_len = each_mem_len + 2
        concat_memory_ids = split_input_ids.reshape(1, mem_len)

        concat_ids = torch.cat([sys_tokens, concat_memory_ids, user_tokens, remaining_ids], dim = 1)
        labels = torch.cat([torch.tensor([[-100] * (sys_len + mem_len + user_len)]), remaining_ids], dim = 1)

        biased_index = []
        bias_position = sys_len
        for _ in range(num_memory):
            biased_index.append([bias_position, bias_position + each_mem_len])
            bias_position = bias_position + each_mem_len

        # attention_matrix = construct_biased_attention_matrix(concat_ids.size(1), biased_index)

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
        text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = text_tokens[:, :self.max_len]
        labels = text_tokens
        # attention_matrix = construct_biased_attention_matrix(text_tokens.size(1) ,[])
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }
    
    def process_textmem(
        self,
        example: Dict[str, str],
    ):
        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")["input_ids"]
    
        input_ids = input_ids[:, :self.max_len - 2] #make space for <begin of text> and <MEM_SUM>
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 2 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]

        # add <|eot_id|>
        # remaining_ids = torch.cat([remaining_ids, torch.tensor([[128009]]).to(remaining_ids.device)], dim=1)
 
        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0))], dim=1)

        mem_len = mem_len + 2 * num_memory
        each_mem_len = each_mem_len + 2
        
        concat_memory_ids = split_input_ids.reshape(1, mem_len)
        
        biased_index = []
        bias_position = 1
        for _ in range(num_memory):
            biased_index.append([bias_position, bias_position + each_mem_len])
            bias_position = bias_position + each_mem_len

        concat_ids = torch.cat([torch.tensor([[128000]]), concat_memory_ids, torch.tensor([[128258]]), remaining_ids], dim = 1)

        labels = torch.cat([torch.tensor([[-100] * (mem_len + 2)]), remaining_ids], dim = 1)
        
        # attention_matrix = construct_biased_attention_matrix(concat_ids.size(1), biased_index)
        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'xsum'
        memory_text = example['document'].split('\n')

        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who helps to summarize the following passages.<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)
        
        memory_ids = [sys_tokens[0]]
        memory_positions = [torch.arange(0,sys_len)]
        current_position = sys_len

        max_memory_length = sys_len
        for idx in range(len(memory_text)):
            text = memory_text[idx]
            memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

            if mem_len > max_memory_length:
                max_memory_length = mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\nSummarize the text provided above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = example['summary'] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'memory_length': max_memory_length,
            'memory_position': memory_positions,
            'split_memory_id': memory_ids,
            'memory_nums': len(memory_text) + 1
            }
    
class bias_reencode_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int
    ) -> None:
        """
        Apply chat template to the conversation between a user and the assistant.
        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
            max_len: the maximum length of the processed text.
            eot_id: the token id of the <turn_end> token.
            prepend_eos: whether prepend an eos token to the processed text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example["conversations"]
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        
        current_position = sys_len
        id_list = [sys_tokens]
        biased_index = []
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
                memory_tokens = self.tokenizer(text, add_special_tokens = False, return_tensors = "pt")['input_ids']
                memory_tokens = torch.cat([torch.tensor([[128256]]), memory_tokens, torch.tensor([[128257]]), torch.tensor([[128258]])], dim = 1)
                id_list.append(memory_tokens)

                mem_len = memory_tokens.size(1)

                biased_index.append([current_position, current_position + mem_len - 1])

                current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        last_q_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        id_list.append(last_q_ids)

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        last_a_ids = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        id_list.append(last_a_ids)    

        concat_ids = torch.cat(id_list, dim =1)

        seq_len = concat_ids.size(1)
        ans_len = last_a_ids.size(1)
        labels = torch.cat([torch.tensor([[-100] * (seq_len - ans_len)]), last_a_ids], dim = 1)

        # attention_matrix = construct_biased_attention_matrix(seq_len, biased_index)
        return {
                'input_ids': concat_ids,
                'labels': labels,
                'biased_index': biased_index
                # 'attention_matrix': attention_matrix
            }    

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
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
            'input_ids': [input_ids_list],
            'labels': [labels],
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }
    
# Todo
    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False, return_tensors= "pt")['input_ids']
        user_len = user_tokens.size(1)

        text = example["text"]
        tokenized = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")
        input_ids = tokenized.input_ids

        # allocate space for "<MEM_SUM><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n". [128009, 128006, 78191, 128007, 271]
        input_ids = input_ids[:, :self.max_len - user_len - sys_len] 
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 3 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]
        ans_len = remaining_ids.size(1)
        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0)), torch.tensor([[128258]] * split_input_ids.size(0))], dim=1)

        mem_len = mem_len + 3 * num_memory
        each_mem_len = each_mem_len + 3
        concat_memory_ids = split_input_ids.reshape(1, mem_len)

        concat_ids = torch.cat([sys_tokens, concat_memory_ids, user_tokens, remaining_ids], dim = 1)
        labels = torch.cat([torch.tensor([[-100] * (sys_len + mem_len + user_len)]), remaining_ids], dim = 1)

        biased_index = []
        bias_position = sys_len
        for _ in range(num_memory):
            biased_index.append([bias_position, bias_position + each_mem_len - 1])
            bias_position = bias_position + each_mem_len

        # attention_matrix = construct_biased_attention_matrix(concat_ids.size(1), biased_index)

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
        text_tokens = self.tokenizer(example["text"], return_tensors= "pt")['input_ids']
        text_tokens = text_tokens[:, :self.max_len]
        labels = text_tokens
        # attention_matrix = construct_biased_attention_matrix(text_tokens.size(1) ,[])
        return {
            'input_ids': text_tokens,
            'labels': labels,
            'biased_index': None
            # 'attention_matrix': attention_matrix
        }
    
    def process_textmem(
        self,
        example: Dict[str, str],
    ):
        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")["input_ids"]
    
        input_ids = input_ids[:, :self.max_len - 1] #make space for <begin of text>
        
        num_memory = random.randint(1, 10)
        each_mem_len = random.randint(50, 150)
        mem_len = num_memory * each_mem_len

        # allocate space for special tokens
        input_len = input_ids.size(1)
        input_ids = input_ids[:, :input_len - 3 * num_memory] 

        memory_ids = input_ids[:, :mem_len]
        remaining_ids = input_ids[:, mem_len:]

        # add <|eot_id|>
        # remaining_ids = torch.cat([remaining_ids, torch.tensor([[128009]]).to(remaining_ids.device)], dim=1)
 
        split_input_ids = memory_ids.reshape(-1, each_mem_len)
        split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0)), torch.tensor([[128258]] * split_input_ids.size(0))], dim=1)

        mem_len = mem_len + 3 * num_memory
        each_mem_len = each_mem_len + 3
        
        concat_memory_ids = split_input_ids.reshape(1, mem_len)
        
        biased_index = []
        bias_position = 1
        for _ in range(num_memory):
            biased_index.append([bias_position, bias_position + each_mem_len - 1])
            bias_position = bias_position + each_mem_len

        concat_ids = torch.cat([torch.tensor([[128000]]), concat_memory_ids, remaining_ids], dim = 1)

        labels = torch.cat([torch.tensor([[-100] * (mem_len + 1)]), remaining_ids], dim = 1)
        
        # attention_matrix = construct_biased_attention_matrix(concat_ids.size(1), biased_index)
        return {
            'input_ids': concat_ids,
            'labels': labels,
            'biased_index': biased_index
            # 'attention_matrix': attention_matrix
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        dataset_id = 'xsum'
        memory_text = example['document'].split('\n')

        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who helps to summarize the following passages.<|eot_id|>"
        sys_tokens = self.tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)
        
        memory_ids = [sys_tokens[0]]
        memory_positions = [torch.arange(0,sys_len)]
        current_position = sys_len

        max_memory_length = sys_len
        for idx in range(len(memory_text)):
            text = memory_text[idx]
            memory_tokens = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

            if mem_len > max_memory_length:
                max_memory_length = mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\nSummarize the text provided above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = self.tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = example['summary'] + "<|eot_id|>"
        answer_tokens = self.tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)

        return {
            'input_ids': remaining_ids,
            'labels': labels,
            'memory_length': max_memory_length,
            'memory_position': memory_positions,
            'split_memory_id': memory_ids,
            'memory_nums': len(memory_text) + 1
            }