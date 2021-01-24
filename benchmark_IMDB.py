import pandas as pd
import argparse
import torch, datasets
import pytorch_lightning as pl

from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler
from bonz_model import BonzClassification, BonzLM, BonzConfig
from torch.cuda.amp import autocast
from pytorch_lightning.loggers import WandbLogger


class Train_Dataset(Dataset):
    def __init__(self, df:datasets.Dataset, tokenizer:BertTokenizerFast):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenizer.encode(self.df['text'][idx], padding='max_length', truncation=True, return_tensors='pt').squeeze(0),
            'labels': torch.tensor(self.df['label'][idx]).long()
        }


class BonzDataModule(pl.LightningDataModule):
    def __init__(self):
        super(BonzDataModule, self).__init__()
        self.tokenzier = BertTokenizerFast.from_pretrained('bert-base-uncased', max_len=512)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_data = datasets.load_dataset('imdb', split='train')
        train_data.set_format('pandas', output_all_columns=True)
        train_dataset = Train_Dataset(train_data, self.tokenzier)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        test_data = datasets.load_dataset('imdb', split='test')
        test_data.set_format('pandas', output_all_columns=True)
        test_dataset = Train_Dataset(test_data, self.tokenzier)
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        return test_dataloader


class BonzModel_PL(pl.LightningModule):
    def __init__(self, config: BonzConfig, **kwargs):
        super(BonzModel_PL, self).__init__()
        self.config = config
        self.kwargs = kwargs
        self.model = BonzClassification(config)

        # model accuracies
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

        # Track train los
        self.train_loss_per_log = 0

    def training_step(self, batch, batch_idx):
        with autocast():
            loss, predicts = self.model(**batch).values()
            self.train_acc.update(predicts.cpu().squeeze(1), batch['labels'].cpu())

        self.log('train/accuracy', self.train_acc.compute(), logger=True, on_epoch=True)

        self.train_loss_per_log += loss.cpu().item()
        self.log('train/loss', self.train_loss_per_log / self.trainer.log_every_n_steps, logger=True, on_step=True)
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.train_loss_per_log = 0

        return loss

    def validation_step(self, batch, batch_idx):
        with autocast():
            loss, predicts = self.model(**batch).values()
            self.val_acc.update(predicts.cpu().squeeze(1), batch['labels'].cpu())

        self.log('val/accuracy', self.val_acc.compute(), logger=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.kwargs['lr'],
            weight_decay=0.001,
            # betas=self.config['betas'],
            # eps=self.config['eps'], weight_decay=self.config['weight_decay']
        )
        return optimizer


if __name__ == '__main__':
    version = 'i2vbk3r3'
    ckpt = torch.load(f'model_weight/LanguamgeModeling/{version}/checkpoints/epoch=0-step=99999.ckpt', map_location='cpu')
    config = BonzConfig(**ckpt['hyper_parameters'])
    config.num_label = 1

    data_module = BonzDataModule()
    model = BonzModel_PL(config=config, lr=5e-5)

    # Load pre-trained weights
    model.load_state_dict(ckpt['state_dict'], strict=False)

    wandb_logger = WandbLogger(name=f'{version}',
                               project='BenchmarkLM',
                               )

    trainer = pl.Trainer(logger=wandb_logger,
                         checkpoint_callback=False,
                         benchmark=True,
                         log_every_n_steps=10,
                         check_val_every_n_epoch=1,
                         #accelerator='ddp',
                         amp_level='native',
                         precision=16,
                         gpus=1,
                         profiler='simple',
                         max_epochs=3,
                         reload_dataloaders_every_epoch=True,
                         #fast_dev_run=True,
                         )

    trainer.fit(model, data_module)







