import transformers
import torch
import time

from torch.utils.data import Dataset, DataLoader, RandomSampler


class BonzDataset(Dataset):
    def __init__(self, data, tokenizer, mlm_prob=0.15):
        self.data = data
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.data)

    def mask_token(self, inputs):
        labels = inputs.clone()

        # Sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)

        # Create special_token matrix
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # Replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx], padding='max_length', return_tensors='pt')
        inputs = {k: inputs[k].squeeze(0) for k in inputs}

        #inputs['input_ids'], inputs['labels'] = self.mask_token(inputs['input_ids'])
        return inputs


class BonzDataCollar():
    def __init__(self, tokenizer, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def mask_token(self, examples):
        inputs = examples['input_ids']
        labels = inputs.clone()

        # Sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)

        # Create special_token matrix
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(label, already_has_special_tokens=True) for label in labels]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # Replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        examples['inputs_ids'] = inputs
        examples['labels'] = labels

        return examples

    def __call__(self, examples):
        return self.mask_token(examples)


