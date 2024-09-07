import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss

class CustomTrainer(Trainer):
    def __init__(self, *args, data_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader

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
        print("Num of data in this batch:", batch_size)
        
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
            mask = mask[505:]

            masked_losses = losses * mask
            final_loss = masked_losses.sum() / mask.sum()
            # print("loss:", final_loss)
            batch_loss += final_loss

        batch_loss /= batch_size
        print("Batch loss:", batch_loss)

        torch.cuda.empty_cache()
        return batch_loss