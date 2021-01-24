import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, RandomSampler


class BonzDataset(Dataset):
    def __init__(self, data, tokenizer,):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode(self.data[idx]['sentences'], padding='max_length', truncation=True,)
        return inputs


class BonzDataCollar():
    def __init__(self, tokenizer, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.all_special_ids = np.array(self.tokenizer.all_special_ids)
        self.mlm_prob = mlm_prob

    def mask_token(self, examples):
        inputs = np.array(examples)
        labels = np.copy(inputs)

        # Sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(inputs.shape, self.mlm_prob)

        # Create special_token matrix
        special_tokens_mask = np.isin(inputs, self.all_special_ids)
        probability_matrix[special_tokens_mask] = 0

        # Create mask indices with binominal
        masked_indices = np.random.binomial(1, probability_matrix).astype(np.bool)
        labels[~masked_indices] = -100

        # Replace masked inputs with masked token ids
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return {'input_ids': torch.tensor(inputs).long(),
                'original_ids': torch.tensor(examples).long(),
                'special_tokens_mask': torch.tensor(special_tokens_mask),
                'positional_ids': torch.arange(512).unsqueeze(0).expand(torch.tensor(inputs).size()),
                'labels': torch.tensor(labels).long()}

    def __call__(self, examples):
        return self.mask_token(examples)

