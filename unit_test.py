import torch, logging, transformers, pandas as pd
#import datasets
from bonz_model import BonzConfig, BonzLM, BonzModelGAN, BonzDataCollar, BonzDataset
#from datasets import load_from_disk

logging.basicConfig(level='DEBUG')


def dummy_dataset():
    data = {
        'sentences': 'This is Duy Duc'
    }
    return [data]*10


def get_data():
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=30)
    #dataset = load_from_disk('/home/bert1130/datasets/bookcorpus')
    dataset = dummy_dataset()
    train_dataset = BonzDataset(dataset, tokenizer)
    data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=2,
                                                   collate_fn=data_collator,
                                                   shuffle=True,
                                                   num_workers=0)
    return train_dataloader


if __name__ == '__main__':
    model = BonzModelGAN(gen_config=BonzConfig(num_layer=2, num_head=2, word_dim=32, emb_dim=16,
                                               seq_len=30, k_dim=20),
                         dis_config=BonzConfig(num_layer=3, num_head=4, word_dim=32, emb_dim=16, num_label=1,
                                               seq_len=30, k_dim=20))
    optimizer = torch.optim.AdamW(model.parameters())
    data_loader = get_data()

    model.train()
    for batch in data_loader:
        outs = model(**batch)
        optimizer.zero_grad()
        outs['gen_loss'].backward()
        optimizer.step()
        break
