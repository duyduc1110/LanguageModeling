import pandas as pd
import argparse
import torch, datasets
import pytorch_lightning as pl
import os

from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler
from bonz_model import BonzClassification, BonzLM, BonzConfig
from torch.cuda.amp import autocast
from pytorch_lightning.loggers import WandbLogger

TASK_NAMES = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']


class Train_Dataset(Dataset):
    def __init__(self, df:datasets.Dataset, tokenizer:BertTokenizerFast):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenizer.encode(self.df[idx]['text'], padding='max_length', truncation=True, return_tensors='pt').squeeze(0),
            'labels': torch.tensor(self.df[idx]['label']).long()
        }


class BonzDataModule(pl.LightningDataModule):
    def __init__(self, task_name=None):
        super(BonzDataModule, self).__init__()
        self.tokenzier = BertTokenizerFast.from_pretrained('bert-base-uncased', max_len=512)
        self.task_name = task_name

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_data = datasets.load_dataset('glue', self.task_name, split='train')
        train_data.rename_column_('sentence', 'text')
        train_dataset = Train_Dataset(train_data, self.tokenzier)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        test_data = datasets.load_dataset('glue', self.task_name, split='validation')
        test_data.rename_column_('sentence', 'text')
        test_dataset = Train_Dataset(test_data, self.tokenzier)
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        return test_dataloader


class BonzModel_PL(pl.LightningModule):
    def __init__(self, config: BonzConfig, task_name, **kwargs):
        super(BonzModel_PL, self).__init__()
        self.config = config
        self.task_name = task_name
        self.kwargs = kwargs
        self.discriminator = BonzClassification(config)

        # task metric
        if task_name in ['sst2']:
            self.train_metric = pl.metrics.Accuracy()
        self.metrics = datasets.load_metric('glue', task_name)

        # Track train los
        self.train_loss_per_log = 0

    def training_step(self, batch, batch_idx):
        with autocast():
            loss, predicts = self.discriminator(**batch).values()
            self.train_metric.update(predicts.cpu().squeeze(1), batch['labels'].cpu())

        self.log('train/accuracy', self.train_metric.compute(), logger=True, on_step=False, on_epoch=True)

        self.train_loss_per_log += loss.cpu().item()
        self.log('train/loss', self.train_loss_per_log / self.trainer.log_every_n_steps, logger=True, on_step=True)
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.train_loss_per_log = 0

        return loss

    def validation_step(self, batch, batch_idx):
        with autocast():
            loss, predicts = self.discriminator(**batch).values()
            self.metrics.add_batch(predictions=(predicts.sigmoid().squeeze(-1)>0.5).long().cpu().tolist(),
                                   references=batch['labels'].cpu().tolist())

        self.log('val/', self.metrics.compute(), logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.kwargs['lr'],
            weight_decay=0.001,
        )
        return optimizer


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--task_name', default='sst2', type=str, help="Set logging level")
    model_parser.add_argument('--weight_path', default='model_weight', type=str, help="Weight path")
    model_parser.add_argument('--project_name', default='LanguamgeModeling', type=str, help="Run name to put in WanDB")
    model_parser.add_argument('--version', default='pa9r1qbn', type=str, help='Run version')

    # Trainer arguments
    model_parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate')
    model_parser.add_argument('--batch_size', default=32, type=int, help='Batch size per device')
    model_parser.add_argument('--log_step', default=500, type=int, help='Steps per log')

    # Add BonzConfig parameter
    cf = BonzConfig()
    for k, v in cf.__dict__.items():
        model_parser.add_argument(f'--{k}', default=v, type=type(v), help=f'{k}')

    args = model_parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Load checkpoint
    ckpt_path = f'{args.weight_path}/{args.project_name}/{args.version}/checkpoints/'
    _, __, ckpt_name = next(iter(os.walk(ckpt_path)))

    # Load config from checkpoint
    ckpt = torch.load(ckpt_path+ckpt_name[0], map_location='cpu')
    config = BonzConfig(**ckpt['hyper_parameters'])
    config.num_label = 1

    # Train and test dataloader
    data_module = BonzDataModule(task_name=args.task_name)

    # Bonz Model with Pytorch-lightning
    model = BonzModel_PL(config=config, task_name=args.task_name, lr=args.lr)

    # Load pre-trained weights
    model.load_state_dict(ckpt['state_dict'], strict=False)

    wandb_logger = WandbLogger(name=f'{args.version}',
                               project='BenchmarkLM',
                               )

    trainer = pl.Trainer(logger=wandb_logger,
                         checkpoint_callback=False,
                         benchmark=True,
                         log_every_n_steps=100,
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







