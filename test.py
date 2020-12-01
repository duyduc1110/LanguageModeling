import transformers
import time
from torch.utils.data import DataLoader, RandomSampler
from utils import BonzDataset, BonzDataCollar


tokenizer = transformers.BertTokenizerFast.from_pretrained('distilbert-base-uncased', model_max_length=256, )
data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)

data = ['This is Duy Duc ne moi nguoi oi la moi nguoi'] * 6400
train_dataset = BonzDataset(data, tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=256,
                              sampler=RandomSampler(train_dataset),
                              collate_fn=data_collator,
                              )

s = time.time()
for a in train_dataloader:
    pass
print('Processing time: ', time.time() - s)