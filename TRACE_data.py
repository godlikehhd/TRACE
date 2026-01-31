from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

class TraceData(Dataset):
    """
    A simplified Dataset class that only loads and provides access to the raw data.
    The heavy lifting of tokenization and padding is offloaded to the collate_fn.
    """
    def __init__(self, data_path):
        # Load the dataset using the efficient datasets library
        self.data = load_dataset("json", data_files=data_path)['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simply return the raw 'messages' field
        return self.data[idx]['messages']


class DataCollatorForChat:
    """
    A collate function that takes a batch of 'messages' lists,
    formats them into strings, and tokenizes them with dynamic padding.
    """
    def __init__(self, tokenizer_name_or_path, max_length=480):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        
        # Set a pad token if the tokenizer doesn't have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        # 'batch' is a list of 'messages' objects from the TraceData dataset
        
        # 1. Apply the chat template to each item in the batch
        formatted_texts = [
            self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=False, 
                tokenize=False
            ) for messages in batch
        ]

        # 2. Tokenize the entire batch of formatted texts at once
        tokenized_batch = self.tokenizer(
            formatted_texts,
            padding=True,          # Pad to the longest sequence in THIS batch
            truncation=True,       # Truncate sequences longer than max_length
            max_length=self.max_length,
            return_tensors="pt"    # Return PyTorch tensors
        )
        return tokenized_batch