
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer


def fetch_data(force):
    # check if data directory exists
    if not Path('data').exists() or force:
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 'feedback-prize-2021'])
        subprocess.run(['unzip', 'feedback-prize-2021.zip', '-d', 'data'])
        print('dataset downloaded')
    else:
        print('dataset already downloaded')

    return pd.read_csv(Path('data', 'train.csv'))

"""
Download writing data
"""
def get_data(test_split_perc: float = 0.2, force: bool = False, limit_data: int = -1, batch_size: int = 32):
    data = fetch_data(force)
    if limit_data > 0:
        data = data[:limit_data]
    train_data = data.sample(frac=1-test_split_perc)
    test_data = data.copy().drop(train_data.index).reset_index(drop=True)

    train_dataset = WritingDataset(train_data)
    test_dataset = WritingDataset(test_data, label_map=train_dataset.label_map)

    return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(test_dataset, batch_size, shuffle=True), train_dataset.label_map


class WritingDataset(Dataset):
    def __init__(self, data: pd.DataFrame, label_map=None):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if label_map is None:
            self.label_map = {
                    label: i 
                    for i, label in enumerate(data['discourse_type'].unique())
                    }
        else:
            self.label_map = label_map
        self.one_hot_encoding = torch.Tensor(np.eye(len(self.label_map)))
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        encoded = self.tokenizer.encode(
                    sample['discourse_text'],
                    is_split_into_words = True,
                    padding = 'max_length',
                    return_tensors = 'pt'
        )[:, :512] # make sure that the encoded is not more than 512 characters

        label = self.label_map[sample['discourse_type']]
        one_hot_label = self.one_hot_encoding[label]

        return (
                encoded.int().squeeze(),
                one_hot_label
            )






if __name__ == '__main__':
    get_data()



