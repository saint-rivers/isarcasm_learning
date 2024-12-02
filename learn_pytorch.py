# -*- coding: utf-8 -*-
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import math

from data import load_data


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Verify idx type
        print(f"Type of idx: {type(idx)}")
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample

data = torch.randn(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 images
labels = torch.randint(0, 10, (100,))  # 100 labels for 10 classes


model_name = "FacebookAI/roberta-base"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
train_tweets, train_labels, test_tweets, test_labels = load_data("soraby")

my_dataset = MyDataset(data, labels)
data_loader = DataLoader(my_dataset, batch_size=4, shuffle=True)

for batch in data_loader:
    print(batch['data'].shape, batch['label'])