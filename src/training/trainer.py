import torch
import matplotlib.pyplot as plt
import json
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from src.utils.cache import generate_kv_with_id, concat_kv, append_kv

class CustomTrainer(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.train_loss_history = []

    def get_train_dataloader(self):
        return self.data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        memory_ids = input_ids[:, 1:501]
        remaining_ids_batch = input_ids[:, 501:]
        # kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]
        split_input_ids = memory_ids.reshape(-1, 50)

        split_past_key_values = generate_kv_with_id(self.model, split_input_ids)
        kv_s = generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)

        num_memory = int((501 - 1) / 50)
        past_key_values_batch = concat_kv(split_past_key_values, num_memory)
        kv_s_concat = append_kv([kv_s] * past_key_values_batch[0][0].size(0),0)
        past_key_values_batch = append_kv([kv_s_concat, past_key_values_batch],2)

        print(remaining_ids_batch.shape, attention_mask.shape, len(past_key_values_batch), past_key_values_batch[0][0].shape)
        outputs = self.model(input_ids=remaining_ids_batch, attention_mask=attention_mask, labels = remaining_ids_batch, past_key_values=past_key_values_batch, use_cache=True)

        logits = outputs.logits  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = remaining_ids_batch[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)
        
        mask = attention_mask[:, 1:].clone()
        mask = mask[:, 501:]
        masked_losses = losses * mask

        masked_losses_sum = masked_losses.sum()
        valid_positions = mask.sum()

        batch_loss = masked_losses_sum / valid_positions
        print(batch_loss)
        # past_key_values_batch = inputs["past_key_values"]

        # # if input_ids is None: 
        # #     return 0

        # batch_loss = 0
        # batch_size = input_ids.size(0)
        # # print("Num of data in this batch:", batch_size)
        
        # # Iterate over each sample in the batch
        # for i in range(batch_size):
        #     input_id = input_ids[i]
        #     attention_msk = attention_mask[i]
        #     past_key_values = past_key_values_batch[i]

        #     outputs = self.model(input_ids=input_id, attention_mask=attention_msk, labels = input_id, past_key_values=past_key_values, use_cache=True)

        #     logits = outputs.logits  

        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = input_id[..., 1:].contiguous()

        #     loss_fct = CrossEntropyLoss(reduction='none')
        #     losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        #     losses = losses.view(shift_logits.size(0), -1)

        #     mask = attention_msk[0, 1:].clone()
        #     mask = mask[505:]
        #     # mask = mask[1001:]
        #     # print(len(mask), losses.size())
        #     masked_losses = losses * mask
        #     final_loss = masked_losses.sum() / mask.sum()
        #     # print("loss:", final_loss)
        #     batch_loss += final_loss

        # batch_loss /= batch_size
        # # print("Batch loss:", batch_loss.item())
        # self.train_loss_history.append(batch_loss.item())
        # torch.cuda.empty_cache()
        return batch_loss

    def save_training_curve(self, output_dir):
        # Save the loss history to a JSON file
        with open(f"{output_dir}/train_loss_history.json", "w") as f:
            json.dump(self.train_loss_history, f)

        # Plot and save the training curve as an image
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(f"{output_dir}/train_loss_curve.png")
        plt.show()