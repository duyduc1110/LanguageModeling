import torch

from utils import BonzDataset, BonzDataCollar
from datasets import load_dataset
from tqdm.auto import tqdm, trange
from transformers import BertTokenizerFast, ReformerConfig, ReformerForMaskedLM
from torch.utils.data import Dataset, DataLoader, RandomSampler
from linformer.linformer import LinformerLM


class BonzDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(BonzDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(self.df[idx]['text'].tolist()[0], padding='max_length', truncation=True,
                                          return_tensors='pt').squeeze(0)
        for i in range(1, 1000):
            concated_text = ' '.join(self.df[idx: idx + i + 1]['text'].tolist())
            temp = self.tokenizer.encode(concated_text, padding='max_length', truncation=True,
                                         return_tensors='pt').squeeze(0)
            if temp[-1] == self.tokenizer.pad_token_id:
                input_ids = temp
            else:
                break
        return {'input_ids': input_ids}


# Load Dataset
dataset = load_dataset('bookcorpus', split='train')
dataset.set_format('pandas')

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('./tokenzier/WordPiece-10k/', max_len=512)

# Load model
model = LinformerLM(10000, 768, 512, 12, heads=12, one_kv_head=True, share_kv=True) # Linformer

# Create train_dataset
train_dataset = BonzDataset(dataset, tokenizer)
data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)
train_dataloader = DataLoader(train_dataset,
                              batch_size=12,
                              sampler=RandomSampler(train_dataset),
                              collate_fn=data_collator
                              )


# Train model
model.cuda()
model.train()
optimizer = torch.optim.AdamW(model.parameters())

for i, inputs in enumerate(train_dataloader):
    optimizer.zero_grad()
    loss = model(inputs['input_ids'].cuda(), inputs['labels'].cuda())['loss']
    loss.backward()
    optimizer.step()

    if i == 100:
        break
