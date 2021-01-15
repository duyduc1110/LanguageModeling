import logging
import os

from bonz_model.utils import BonzDataset, BonzDataCollar
from datasets import load_from_disk
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from bonz_model.linformer import LinformerLM


logging.basicConfig(level='INFO')
os.environ['WANDB_WATCH '] = 'all'

BATCH_SIZE = 96
MAX_STEP = int(100000 * 256 / (BATCH_SIZE * 3))
WARMUP_STEP = int(MAX_STEP/10)
LEARNING_RATE = 2e-4

if __name__ == '__main__':
    # Load Dataset
    dataset = load_from_disk('/home/bert1130/datasets/bookcorpus')

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('/home/bert1130/Github/test/tokenizer/WordPiece-10k', max_len=512)

    # Load model
    model = LinformerLM(tokenizer.vocab_size, dim=768, seq_len=512, depth=12, k=256,
                        heads=12, dim_head=None, one_kv_head=True, share_kv=True, reversible=False)
    model.cuda()
    model.train()

    '''
    # Create optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=MAX_STEP,
                                                    pct_start=0.05, div_factor=10, final_div_factor=100)
    '''

    # Create train_dataset
    train_dataset = BonzDataset(dataset, tokenizer)
    data_collator = BonzDataCollar(tokenizer, mlm_prob=0.15)
    '''
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  sampler=RandomSampler(train_dataset),
                                  num_workers=0,
                                  collate_fn=data_collator,
                                  )
    '''

    training_args = TrainingArguments(
        output_dir="./BonzLM",
        overwrite_output_dir=True,
        # Training step
        max_steps=MAX_STEP,
        warmup_steps=WARMUP_STEP,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=10000,
        save_total_limit=2,
        # mixed precision
        fp16=True,
        fp16_opt_level='O2',
        seed=42,
        # Learning rate setup
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        #max_grad_norm=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        #optimizers=(optimizer, scheduler)
    )

    trainer.train()



    '''
    s_total = time.time()
    s = time.time()
    for idx, a in enumerate(train_dataloader):
        print('Load data time: ', time.time()-s)
        s = time.time()
        a = torch.tensor(a[0])
        print(a.shape)

        optimizer.zero_grad()
        loss = model(a.cuda())['logits']
        loss.mean().backward()
        optimizer.step()
        print('Running time: ', time.time() - s)
        s = time.time()
        print(torch.cuda.memory_reserved(), '\n--------------------------------')

        if idx == 10:
            print('Totall time: ', time.time()-s_total)
            break
    '''



