# import random

# def split_list_randomly(lst, k):
#     N = len(lst)
#     if k <= 0 or k > N:
#         raise ValueError("k must be between 1 and the length of the list")
#     # Generate k-1 unique random breakpoints between 1 and N-1
#     breaks = sorted(random.sample(range(1, N), k - 1))
#     # Add the start and end points
#     breaks = [0] + breaks + [N]
#     # Compute the sizes of each sublist
#     sizes = [breaks[i+1] - breaks[i] for i in range(k)]

#     print(sizes)
#     # Split the list according to the sizes
#     result = []
#     index = 0
#     for size in sizes:
#         result.append(lst[index:index + size])
#         index += size
#     return result

# # Example usage:
# lst = list(range(1, 21))  # A list with 20 items
# k = 5  # Number of sublists
# random_sublists = split_list_randomly(lst, k)
# print(random_sublists)


#         text = example["text"]
#         input_ids = self.tokenizer(text, add_special_tokens= False, return_tensors= "pt")["input_ids"]
    
#         input_ids = input_ids[:, :self.max_len - 2] #make space for <begin of text> and <MEM_SUM>
        
#         num_memory = random.randint(1, 10)
#         each_mem_len = random.randint(50, 150)
#         mem_len = num_memory * each_mem_len

#         # allocate space for special tokens
#         input_len = input_ids.size(1)
#         input_ids = input_ids[:, :input_len - 2 * num_memory] 

#         memory_ids = input_ids[:, :mem_len]
#         remaining_ids = input_ids[:, mem_len:]
 
#         split_input_ids = memory_ids.reshape(-1, each_mem_len)
#         split_input_ids = torch.cat([torch.tensor([[128256]] * split_input_ids.size(0)), split_input_ids, torch.tensor([[128257]] * split_input_ids.size(0))], dim=1)

#         mem_len = mem_len + 2 * num_memory
#         each_mem_len = each_mem_len + 2
        
#         concat_memory_ids = split_input_ids.reshape(1, mem_len)
        
#         biased_index = []
#         bias_position = 1
#         for _ in range(num_memory):
#             biased_index.append([bias_position, bias_position + each_mem_len])
#             bias_position = bias_position + each_mem_len

#         concat_ids = torch.cat([torch.tensor([[128000]]), concat_memory_ids, torch.tensor([[128258]]), remaining_ids], dim = 1)

#         labels = torch.cat([torch.tensor([[-100] * (mem_len + 2)]), remaining_ids], dim = 1)

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

print(t("Hello world", add_special_tokens= False)['input_ids'][:50])
