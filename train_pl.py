import pytorch_lightning as pl, torch.nn as nn, pandas as pd
import torch
import transformers
import argparse
import logging

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from linformer.linformer import LinformerLM
from transformers import BertTokenizerFast
from utils import BonzDataset, BonzDataCollar
from datasets import load_from_disk, load_dataset
from torch.cuda.amp import autocast, GradScaler


class LinformerLM_PL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LinformerLM(**config)
        self.save_hyperparameters(config)
        self.train_loss_per_log = 0

    def training_step(self, batch, batch_idx):
        with autocast():
            loss = self.model(**batch)['loss']

        # Logging when training
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = (
            running_train_loss.cpu().item()
            if running_train_loss is not None
            else float("NaN")
        )
        self.log('train/loss_mean', avg_training_loss, logger=True, on_step=True)
        self.log('num_samples', (batch_idx + 1) * batch['x'].shape[0] * 3, logger=True, on_step=True)

        self.train_loss_per_log += loss.cpu().item()
        self.log('train/loss', self.train_loss_per_log/self.trainer.log_every_n_steps, logger=True, on_step=True)
        if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
            self.train_loss_per_log = 0

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=0.001
                                      # betas=self.config['betas'],
                                      # eps=self.config['eps'], weight_decay=self.config['weight_decay']
                                      )

        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                                         self.config['lr'],
                                                                         total_steps=self.config['training_step'],
                                                                         pct_start=0.05,
                                                                         anneal_strategy='cos',
                                                                         cycle_momentum=True,
                                                                         base_momentum=0.85,
                                                                         max_momentum=0.95,
                                                                         div_factor=10.0,
                                                                         final_div_factor=20.0),
                        'name': 'train/learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model arguments
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('--model_path', default=None, type=str, help="Model path")
    model_parser.add_argument('--remote', default=True, type=bool, help="Run remote or local")
    model_parser.add_argument('--tokenizer_path', default='bert-base-uncased', type=str, help="Tokenizer path")
    model_parser.add_argument('--seq_len', default=512, type=int, help="Max sequence length")
    model_parser.add_argument('--depth', default=12, type=int, help="Number of layers")
    model_parser.add_argument('--embedding_dim', default=768, type=int, help='Embedding dim', dest='dim')

    # Trainer arguments
    model_parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    model_parser.add_argument('--training_step', default=100000, type=int, help='Training steps')
    model_parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    model_parser.add_argument('--log_step', default=500, type=int, help='Steps per log')

    args = model_parser.parse_args()
    return args


def get_tokenizer(args):
    return BertTokenizerFast.from_pretrained(args.tokenizer_path, max_len=args.seq_len)


def get_model(args, tokenizer):
    config = vars(args)
    config['num_tokens'] = tokenizer.vocab_size
    # logger.info(config)
    model = LinformerLM_PL(config)
    # logger.info('Model created')
    return model


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    # logging.basicConfig(level=args.logging_level)
    # logger = logging.getLogger('model')

    # Get tokenizer
    tokenizer = get_tokenizer(args)

    # Init model
    model = get_model(args, tokenizer)

    # Get data
    if args.remote is True:
        dataset = load_from_disk('/home/bert1130/datasets/bookcorpus')
    else:
        dataset = load_dataset('bookcorpus', split='train')
        dataset.rename_column_('text', 'sentences')
    train_dataset = BonzDataset(dataset, tokenizer)
    data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   collate_fn=data_collator,
                                                   shuffle=True,
                                                   num_workers=0)

    # Define logger
    wandb_logger = WandbLogger(name=f'{tokenizer.vocab_size}_{args.lr:.0e}OneCycleLR_batch{args.batch_size}',
                               project='LanguamgeModeling',
                               )

    # Define callbacks:
    lr_monitor = LearningRateMonitor('step')

    # Create trainer
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=[lr_monitor],
                         log_every_n_steps=args.log_step,
                         accelerator='ddp',
                         amp_level='native',
                         precision=16,
                         gpus=-1,
                         profiler='simple',
                         weights_save_path='./model_weight',
                         max_steps=args.training_step,
                         #plugins=['ddp_sharded'],
                         #weights_summary='full'
                         )

    trainer.fit(model, train_dataloader)
