import math
import torch
from torch import nn
import logging
import json


model_logger = logging.getLogger('Linformer-logger')
torch.random.manual_seed(42)


# helper functions
def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class BonzConfig(object):
    def __init__(self, seq_len=512, word_dim=768, emb_dim=768, k_dim=256, one_project=True, dropout=0.1, group_att=4, group_ff=4,
                 ff_mul=2, kernel_att=1, kernel_ff=1, num_layer=12, num_head=12, vocab_size=30522, num_label=2,
                 eps=1e-12, glu=True, **kwargs):
        self.seq_len = seq_len
        self.word_dim = word_dim
        self.emb_dim = emb_dim
        self.k_dim = k_dim
        self.one_project = one_project
        self.ff_mul = ff_mul
        self.num_layer = num_layer
        self.num_head = num_head
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.kernel_att = kernel_att
        self.kernel_ff = kernel_ff
        self.group_att = group_att
        self.group_ff = group_ff
        self.eps = eps
        self.glu = glu
        self.num_label = num_label
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__} {json.dumps(self.__dict__, indent=2)}'



class BonzEmbedding(nn.Module):
    def __init__(self, config: BonzConfig):
        super(BonzEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.word_dim, padding_idx=0)
        self.positional_embedding = nn.Embedding(config.seq_len, config.word_dim, padding_idx=0)
        if config.emb_dim != config.word_dim:
            self.scale = nn.Linear(config.word_dim, config.emb_dim)
        else:
            self.scale = None
        self.norm = nn.LayerNorm(config.emb_dim, config.eps)

    def forward(self, input_ids, positional_ids):
        token_embeddings = self.token_embedding(input_ids)

        if positional_ids is None:
            batch, seq_len = input_ids.size()
            positional_ids = torch.arange(seq_len, device=input_ids.device).repeat(batch).reshape(batch, seq_len)
        positional_embeddings = self.positional_embedding(positional_ids)
        out_embeddings = token_embeddings + positional_embeddings
        if self.scale is not None:
            out_embeddings = self.scale(out_embeddings)
        out_embeddings = self.norm(out_embeddings)
        return out_embeddings


class BonzSelfAttention(nn.Module):
    def __init__(self, config: BonzConfig, project_k: nn.Parameter = None):
        super().__init__()
        assert (config.emb_dim % config.num_head) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = config.seq_len
        self.k_dim = config.k_dim

        self.num_head = config.num_head
        self.dim_head = config.emb_dim // config.num_head

        ''' Linear Attention
        self.query = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.key = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.value = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        '''

        ''' CNN Attention '''
        self.query = nn.Conv1d(config.emb_dim,
                               config.emb_dim,
                               kernel_size=config.kernel_att,
                               padding=int(config.kernel_att / 2),
                               groups=config.group_att,
                               bias=False)
        self.key = nn.Conv1d(config.emb_dim,
                             config.emb_dim,
                             kernel_size=config.kernel_att,
                             groups=config.group_att,
                             padding=int(config.kernel_att / 2),
                             bias=False)
        self.value = nn.Conv1d(config.emb_dim,
                               config.emb_dim,
                               kernel_size=config.kernel_att,
                               groups=config.group_att,
                               padding=int(config.kernel_att / 2),
                               bias=False)

        self.project_k = project_k if project_k is not None else nn.Parameter(init_(torch.rand(config.seq_len,
                                                                                               config.k_dim)))
        self.dropout = nn.Dropout(config.dropout)

        self.output = nn.Linear(config.emb_dim, config.emb_dim)
        #self.norm = nn.LayerNorm(config.emb_dim, eps=config.eps)

    def forward(self, input_embedding, **kwargs):
        batch, seq_len, embed_dim = input_embedding.shape
        head_dim, num_head, key_dim = self.dim_head, self.num_head, self.k_dim

        # Transormer input_embedding for Conv1d
        input_embedding = input_embedding.permute(0, 2, 1)

        # Calculate query vectors
        queries = self.query(input_embedding).permute(0, 2, 1)

        # Projecting keys vectors
        keys = self.key(input_embedding).permute(0, 2, 1)
        keys = torch.einsum('bnd, nk -> bkd', keys, self.project_k).permute(0, 2, 1)

        # Projecting values vectors
        values = self.key(input_embedding).permute(0, 2, 1)
        values = torch.einsum('bnd, nk -> bkd', values, self.project_k)

        # Reshape Q K V
        queries = queries.view(batch, seq_len, num_head, head_dim).transpose(1, 2)
        keys = keys.view(batch, key_dim, num_head, head_dim).transpose(1, 2)
        values = values.view(batch, key_dim, num_head, head_dim).transpose(1, 2)

        # Self-attention score
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) / (head_dim ** 0.5)
        attention_score = dots.softmax(dim=-1)
        attn = self.dropout(attention_score)
        attn = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # Merge heads & layer norm
        out = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        #out = self.norm(input_embedding.permute(0, 2, 1) + self.output(out))
        return out


class BonzFeedForward(nn.Module):
    def __init__(self, config: BonzConfig, activation=None, **kwargs):
        super().__init__()
        activation = default(activation, nn.GELU)
        self.glu = config.glu
        self.w1 = nn.Conv1d(config.emb_dim,
                            config.emb_dim * config.ff_mul * 2 if config.glu else config.emb_dim * config.ff_mul,
                            kernel_size=config.kernel_ff,
                            groups=config.group_ff,
                            padding=int(config.kernel_ff/2))
        self.act = activation()
        self.dropout = nn.Dropout(config.dropout)
        self.w2 = nn.Conv1d(config.emb_dim * config.ff_mul, config.emb_dim,
                            kernel_size=config.kernel_ff,
                            groups=config.group_ff,
                            padding=int(config.kernel_ff/2))

    def forward(self, input_embedding: torch.FloatTensor, **kwargs):
        if not self.glu:
            x = self.act(self.w1(input_embedding.permute(0,2,1)))
        else:
            x, v = self.w1(input_embedding.permute(0,2,1)).chunk(2, dim=1)
            x = self.act(x) * v

        x = self.dropout(x)
        out = self.w2(x)
        return out.permute(0,2,1)

class ReZero(nn.Module):
    def __init__(self):
        super(ReZero, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, function):
        return inputs + self.alpha * function


class BonzEncoderLayer(nn.Module):
    def __init__(self, config:BonzConfig, **kwargs):
        super().__init__()
        self.attention_layer = BonzSelfAttention(config=config, **kwargs)
        self.rezero_att = ReZero()
        self.feed_forward = BonzFeedForward(config=config, **kwargs)
        self.rezero_ff = ReZero()

    def forward(self, inputs):
        # Attention layer
        att = self.attention_layer(inputs)
        att_out = self.rezero_att(inputs, att)

        # Feed forward layer
        ff = self.feed_forward(att_out)
        ff_out = self.rezero_ff(att_out, ff)

        return ff_out


class BonzCoreModel(nn.Module):
    def __init__(self, config: BonzConfig, **kwargs):
        super(BonzCoreModel, self).__init__()
        self.embedding_layers = BonzEmbedding(config=config)
        if config.one_project:
            self.K_shared = nn.Parameter(init_(torch.zeros(config.seq_len, config.k_dim)))
            self.encoder_layers = nn.ModuleList([BonzEncoderLayer(config=config, project_k=self.K_shared) for i in range(config.num_layer)])
        else:
            self.encoder_layers = nn.ModuleList([BonzEncoderLayer(config=config) for i in range(config.num_layer)])

    def forward(self, input_ids, positional_ids):
        outputs = self.embedding_layers(input_ids, positional_ids)
        for layer in self.encoder_layers:
            outputs = layer(outputs)
        return outputs


class BonzLMHead(nn.Module):
    def __init__(self, config: BonzConfig):
        super(BonzLMHead, self).__init__()
        self.to_logits = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, inputs):
        return self.to_logits(inputs)


class BonzClassificationHead(nn.Module):
    def __init__(self, config: BonzConfig):
        super(BonzClassificationHead, self).__init__()
        self.pooler = nn.Linear(config.emb_dim, config.emb_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.to_logits = nn.Linear(config.emb_dim, config.num_label)

    def forward(self, inputs):
        outs = self.activation(self.pooler(inputs))
        outs = self.dropout(outs)
        return self.to_logits(outs)


class BonzBaseModel(nn.Module):
    def __init__(self):
        super(BonzBaseModel, self).__init__()
        
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for k, p in self.named_parameters():
            if 'project_k' not in k and p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        model_logger.info('Sucessful init weight with xavier')


class BonzLM(BonzBaseModel):
    def __init__(self, config: BonzConfig, **kwargs):
        super(BonzLM, self).__init__()
        self.core = BonzCoreModel(config)
        self.lm_head = BonzLMHead(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self._reset_parameters()

    def forward(self, input_ids, positional_ids=None, labels=None):
        attention_out = self.core(input_ids, positional_ids)
        logits = self.lm_head(attention_out)

        if labels is not None:
            masked_lm_loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            model_logger.debug(f'loss = {masked_lm_loss}')
            return {'loss': masked_lm_loss, 'logits': logits}
        else:
            return {'logits': logits}


class BonzClassification(BonzBaseModel):
    def __init__(self, config: BonzConfig, **kwargs):
        super(BonzClassification, self).__init__()
        self.core = BonzCoreModel(config)
        self.cls_head = BonzClassificationHead(config)

        self.num_label = config.num_label
        self.loss_fn = nn.CrossEntropyLoss() if config.num_label > 1 else nn.BCEWithLogitsLoss()

        self._reset_parameters()

    def forward(self, input_ids, positional_ids=None, labels=None):
        attention_out = self.core(input_ids, positional_ids)
        cls_tokens = attention_out[:, 0, :]
        logits = self.cls_head(cls_tokens)

        if labels is not None:
            loss = self.loss_fn(
                logits if self.num_label > 1 else logits.view(-1),
                labels if self.num_label > 1 else labels.float()
            )
            model_logger.debug(f'loss = {loss}')
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}


class BonzDiscriminator(BonzBaseModel):
    def __init__(self, config: BonzConfig, **kwargs):
        super(BonzDiscriminator, self).__init__()
        self.core = BonzCoreModel(config)
        self.cls_head = BonzClassificationHead(config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._reset_parameters()

    def forward(self, input_ids, positional_ids=None, special_tokens_mask=None, labels=None):
        attention_out = self.core(input_ids, positional_ids)
        logits: torch.FloatTensor = self.cls_head(attention_out)

        if labels is not None:
            loss = self.loss_fn(logits[~special_tokens_mask].view(-1), labels[~special_tokens_mask].view(-1).float())
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}


## GAN PRE-TRAINING TECHNIQUE FROM ELECTRA
class BonzModelGAN(BonzBaseModel):
    def __init__(self, gen_config: BonzConfig, dis_config: BonzConfig, **kwargs):
        super(BonzModelGAN, self).__init__()
        self.generator = BonzLM(gen_config, **kwargs)
        self.discriminator = BonzDiscriminator(dis_config, **kwargs)
        self.tie_embedding()

    def forward(self, input_ids=None, original_ids=None, special_tokens_mask=None, positional_ids=None, labels=None):
        gen_loss, gen_logits = self.generator(input_ids, positional_ids, labels).values()

        # Get Discriminator input labels
        gen_predict = gen_logits.softmax(-1).argmax(-1).long()  # Predictions from Generator
        dis_labels = self.generate_labels(input_ids, gen_predict, original_ids)

        dis_loss, dis_logits = self.discriminator(original_ids, positional_ids, special_tokens_mask, dis_labels).values()

        return {
            'gen_loss': gen_loss,
            'gen_logits': gen_logits,
            'dis_loss': dis_loss,
            'dis_logits': dis_logits
        }

    def tie_embedding(self):
        for (k_gen, v_gen), (k_dis, v_dis) in zip(self.generator.core.embedding_layers.named_parameters(),
                                                  self.discriminator.core.embedding_layers.named_parameters()):
            if 'token' in k_gen or 'position' in k_gen:
                v_gen = v_dis
                model_logger.debug(f'Tie weight between {k_gen} and {k_dis}')

    def generate_labels(self, input_ids: torch.Tensor, predicts: torch.Tensor, original_ids):
        temp_inputs = input_ids.clone()
        temp_predicts = predicts.clone()
        temp_inputs[input_ids == 103] = temp_predicts[input_ids == 103]  # Map generator predictions to input_ids
        dis_labels = (temp_inputs == original_ids).float()  # Labels for the discriminator
        return dis_labels
