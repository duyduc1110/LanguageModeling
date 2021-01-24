import torch, logging, transformers, datasets
from bonz_model import BonzConfig, BonzLM, BonzModelGAN, BonzDataCollar, BonzDataset
from datasets import load_from_disk

logging.basicConfig(level='DEBUG')


def get_data():
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    dataset = load_from_disk('/home/bert1130/datasets/bookcorpus')
    train_dataset = BonzDataset(dataset, tokenizer)
    data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=2,
                                                   collate_fn=data_collator,
                                                   shuffle=True,
                                                   num_workers=0)
    return train_dataloader


if __name__ == '__main__':
    model = BonzModelGAN(gen_config=BonzConfig(num_layer=4, num_head=4, word_dim=768, emb_dim=256),
                         dis_config=BonzConfig(word_dim=768, emb_dim=768, num_label=1))
    optimizer = torch.optim.AdamW(model.parameters())
    data_loader = get_data()

    model.train()
    for batch in data_loader:
        outs = model(**batch)
        optimizer.zero_grad()
        outs['gen_loss'].backward()
        optimizer.step()
        break
