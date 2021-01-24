import pytorch_lightning as pl
import torch
import argparse
import logging

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from bonz_model import BonzConfig, BonzLM, BonzClassification, BonzModelGAN
from transformers import BertTokenizerFast
from bonz_model.utils import BonzDataset, BonzDataCollar
from datasets import load_from_disk, load_dataset
from torch.cuda.amp import autocast

torch.random.manual_seed(42)

'''
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=True,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
'''


class PrintOutModelCB(pl.Callback):
    def __init__(self):
        super(PrintOutModelCB, self).__init__()

    def on_train_start(self, trainer, pl_module, *args, **kwargs):
        logger.info(pl_module.model)
        logger.info('Start training!!!')


class BonzDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(BonzDataModule, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', max_len=512)
        self.args = args

    def train_dataloader(self, **kwargs):
        dataset = load_from_disk('/home/bert1130/datasets/bookcorpus/')
        train_dataset = BonzDataset(dataset, self.tokenizer)
        data_collator = BonzDataCollar(self.tokenizer, mlm_prob=0.15)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.args.batch_size,
                                                       collate_fn=data_collator,
                                                       shuffle=True,
                                                       num_workers=0)
        return train_dataloader


class BonzLM_PL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = BonzModelGAN(gen_config=BonzConfig(num_layer=4, num_head=4, emb_dim=256),
                                  dis_config=BonzConfig(**config))

        self.save_hyperparameters(config)
        self.train_loss_per_log = 0

    def training_step(self, batch, batch_idx, optimizer_idx):
        with autocast():
            outs = self.model(**batch)
            gen_loss = outs['gen_loss']
            dis_loss = outs['dis_loss']

        (gen_opt, dis_opt) = self.optimizers()

        # Manual backward Generator loss
        self.manual_backward(gen_loss, gen_opt)
        gen_opt.step()

        # Manual backward Discriminator loss
        self.manual_backward(dis_loss, dis_opt)
        dis_opt.step()

        # Logging when training
        self.log('num_samples', (self.global_step + 1) * batch['input_ids'].shape[0] * 3, logger=True, on_step=True)

        total_loss = gen_loss + dis_loss
        self.train_loss_per_log += total_loss.cpu()
        self.log('train/loss', self.train_loss_per_log/self.trainer.log_every_n_steps, logger=True, on_step=True)
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.train_loss_per_log = 0

        self.log('gen_loss', gen_loss, logger=True, on_step=True)
        self.log('dis_loss', dis_loss, logger=True, on_step=True)
        self.log('l', total_loss, prog_bar=True, logger=False, on_step=True)

    def configure_optimizers(self):
        gen_optimizer = torch.optim.AdamW(self.model.generator.parameters(), lr=self.config['lr'], weight_decay=0.001)
        dis_optimizer = torch.optim.AdamW(self.model.discriminator.parameters(), lr=self.config['lr'], weight_decay=0.001)

        gen_lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(gen_optimizer,
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

        dis_lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(dis_optimizer,
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

        return [gen_optimizer, dis_optimizer], [gen_lr_scheduler, dis_lr_scheduler]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        items.pop('v_num', None)
        return items


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('--model_path', default=None, type=str, help="Model path")
    model_parser.add_argument('--remote', default=True, type=bool, help="Run remote or local")
    model_parser.add_argument('--tokenizer_path', default='bert-base-uncased', type=str, help="Tokenizer path")

    # Trainer arguments
    model_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    model_parser.add_argument('--training_step', default=100000, type=int, help='Training steps')
    model_parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    model_parser.add_argument('--log_step', default=500, type=int, help='Steps per log')

    # Add BonzConfig parameter
    cf = BonzConfig()
    for k, v in cf.__dict__.items():
        model_parser.add_argument(f'--{k}', default=v, type=type(v), help=f'{k}')

    args = model_parser.parse_args()
    return args


def get_tokenizer(args):
    return BertTokenizerFast.from_pretrained(args.tokenizer_path, max_len=args.seq_len)


def get_model(args, tokenizer):
    config = vars(args)
    config['num_label'] = 1
    model = BonzLM_PL(config)

    return model


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')

    # Get tokenizer
    tokenizer = get_tokenizer(args)

    # Init model
    model = get_model(args, tokenizer)

    # Get data
    data_module = BonzDataModule(args)

    # Define logger
    wandb_logger = WandbLogger(name=f'BonzGAN_{tokenizer.vocab_size}_{args.lr:.0e}OneCycleLR_batch{args.batch_size}',
                               project='LanguamgeModeling',
                               )

    # Define callbacks:
    lr_monitor = LearningRateMonitor('step')

    # Create trainer
    trainer = pl.Trainer(logger=wandb_logger,
                         gradient_clip_val=0.5,
                         automatic_optimization=False,
                         callbacks=[lr_monitor, PrintOutModelCB()],
                         benchmark=True,
                         log_every_n_steps=args.log_step,
                         accelerator='ddp',
                         amp_level='native',
                         precision=16,
                         gpus=-1,
                         profiler='simple',
                         weights_save_path='./model_weight',
                         max_steps=args.training_step,
                         reload_dataloaders_every_epoch=True,
                         #plugins=['ddp_sharded'],
                         weights_summary='full',
                         )

    trainer.fit(model, data_module)

