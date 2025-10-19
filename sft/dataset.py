import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MultipleKnowledgeDataset(Dataset):
    def __init__(self,train_file,pretrained_model,**kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.data_prompts_list = torch.load(train_file)
    def __getitem__(self, index):
        return self.data_prompts_list[index]
    def __len__(self,):
        return len(self.data_prompts_list)
    def collate_fn(self,batch):
        input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask_list = [torch.tensor(item["attention_mask"]) for item in batch]
        labels_list = [torch.tensor(item["labels"]) for item in batch]
        
        # 使用 pad_sequence 自动 padding，padding_value 指定填充值
        input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels_batch = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        re_dict = {
            "input_ids":input_ids_batch,
            "attention_mask":attention_mask_batch,
            "labels":labels_batch
        }
        return re_dict

