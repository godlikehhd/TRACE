from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import json
import os
import numpy as np
import random
import time
import datetime
import logging
import traceback

class TraceData(Dataset):
    def __init__(self, data_path, model_name_or_path):
        self.data = load_dataset("json", data_files=data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        messages = data['messages']
        input = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, max_length=4096, return_tensors='pt', padding=True, truncation=True, padding_value=self.tokenizer.pad_token_id)
        return input
    
    def collate_fn(self, batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        return input_ids, attention_mask









if __name__ == "__main__":
    main()