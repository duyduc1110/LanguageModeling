import time
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm, trange
from transformers import BertTokenizerFast, ReformerConfig, ReformerForMaskedLM
from torch.utils.data import Dataset, DataLoader, RandomSampler


class BonzDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(BonzDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(self.df.loc[idx, 'text'], padding='max_length', truncation=True,)
        for i in range(1, 1000):
            concated_text = ' '.join(df.loc[idx: idx + i + 1, 'text'].tolist())
            temp = self.tokenizer.encode(concated_text, padding='max_length', truncation=True,)
            if temp[-1] == self.tokenizer.pad_token_id:
                input_ids = temp
            else:
                break
        return input_ids


# Load Dataset
dataset = load_dataset('bookcorpus', split='train')
dataset.set_format('pandas')

dataset.save_to_disk()

df = pd.read_csv('bookcorpus.txt', names=['text'])

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('./tokenzier/WordPiece-10k/', max_len=512)

# Create train_dataset
train_dataset = BonzDataset(df, tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=256,
                              shuffle=True
                              )

s = time.time()
inputs = {'input_ids': []}
for idx, a in tqdm(enumerate(train_dataloader)):
    inputs['input_ids'].extend(a)
    if idx == 10:
        break
print(time.time()-s)

