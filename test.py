import transformers
import time
import torch
import os
from torch.utils.data import DataLoader, RandomSampler
from utils import BonzDataset, BonzDataCollar
from linformer.linformer import LinformerLM


if __name__ == '__main__':
    tokenizer = transformers.BertTokenizerFast.from_pretrained('distilbert-base-uncased', model_max_length=512, )
    data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)

    data = ['This is Duy Duc ne moi nguoi oi la moi nguoi'] * 6400
    train_dataset = BonzDataset(data, tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=256,
                                  #sampler=RandomSampler(train_dataset),
                                  #collate_fn=data_collator,
                                  )

    s = time.time()
    for a in train_dataloader:
        data_collator(a)
        print(time.time()-s)
        print(a['input_ids'].shape)
        break

    #model = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased') # BERT
    model = LinformerLM(10000, 768, 512, 12, heads=12, one_kv_head=True, share_kv=True) # Linformer

    model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(),)

    batch = 11
    inputs = torch.randint(10000, (batch, 512)).cuda()
    for _ in range(10):
        optimizer.zero_grad()
        #outputs = model(inputs.cuda(), return_dict=False)[0] # BERT
        outputs = model(inputs.cuda()) # Linformer
        print(outputs.shape)
        print(outputs.mean().shape)
        outputs.mean().backward()
        optimizer.step()

