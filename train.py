import torch

from bonz_model.utils import BonzDataset, BonzDataCollar
from datasets import load_dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, RandomSampler
from bonz_model.linformer import LinformerLM


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
