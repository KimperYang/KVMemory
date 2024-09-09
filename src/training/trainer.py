import torch
import matplotlib.pyplot as plt
import json
from transformers import Trainer
from torch.nn import CrossEntropyLoss

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
        past_key_values_batch = inputs["past_key_values"]

        # if input_ids is None: 
        #     return 0

        batch_loss = 0
        batch_size = input_ids.size(0)
        # print("Num of data in this batch:", batch_size)
        
        # Iterate over each sample in the batch
        for i in range(batch_size):
            input_id = input_ids[i]
            attention_msk = attention_mask[i]
            past_key_values = past_key_values_batch[i]

            outputs = self.model(input_ids=input_id, attention_mask=attention_msk, labels = input_id, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_id[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            mask = attention_msk[0, 1:].clone()
            # mask = mask[505:]
            mask = mask[1001:]
            # print(len(mask), losses.size())
            masked_losses = losses * mask
            final_loss = masked_losses.sum() / mask.sum()
            # print("loss:", final_loss)
            batch_loss += final_loss

        batch_loss /= batch_size
        # print("Batch loss:", batch_loss.item())
        self.train_loss_history.append(batch_loss.item())
        torch.cuda.empty_cache()
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