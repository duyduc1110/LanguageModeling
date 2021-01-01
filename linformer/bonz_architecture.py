import math
import torch
from torch import nn
import torch.nn.functional as F
import logging

from linformer.reversible import ReversibleSequence, SequentialSequence


model_logger = logging.getLogger('Linformer-logger')
GELU = nn.GELU


# helper functions
def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class BonzConfig():
    def __init__(self, seq_len=512, emb_dim=768, k_dim=256, drop_out=0.1,
                 ff_mul=4, kernel_att=1, kernel_ff=1,
                 eps=1e-12, glu=False):
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.k_dim = k_dim
        self.ff_mul = ff_mul
        self.dropout = drop_out
        self.kernel_att = kernel_att
        self.kernel_ff = kernel_ff
        self.eps = eps
        self.glu = glu


# helper classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim, 1e-12, True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class FeedForward(nn.Module):
    def __init__(self, config: BonzConfig, dim, mult=4, dropout=0.1, activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)
        self.glu = config.glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.w1 = nn.Conv1d(config.emb_dim,
                            config.emb_dim * config.ff_mul * 2 if config.glu else config.emb_dim * config.ff_mul,
                            kernel_size=config.kernel_ff,
                            padding=int(config.kernel_ff/2))
        self.act = activation()
        self.dropout1 = nn.Dropout(dropout)
        self.w2 = nn.Conv1d(config.emb_dim * config.ff_mul, config.emb_dim,
                            kernel_size=config.kernel_ff,
                            padding=int(config.kernel_ff/2))
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.act(self.w1(x))
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.norm(self.w2(x))
        return x


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.1):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads
        self.dim_head = default(dim_head, dim // heads)

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None, **kwargs):
        batch, seq_len, embed_dim = x.shape
        head_dim, num_head, key_dim = self.dim_head, self.heads, self.k

        kv_len = seq_len if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(batch, seq_len, num_head, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(batch, key_dim, -1, head_dim).transpose(1, 2).expand(-1, num_head, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (head_dim ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.to_out(out)


class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False,
                 reversible=False, dropout=0.1):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head,
                                          share_kv=share_kv, dropout=dropout)
            ff = FeedForward(dim)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x):
        return self.net(x)


class LinformerLM(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False,
                 share_kv=False, reversible=False, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(dim, seq_len, depth, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head,
                                   share_kv=share_kv, reversible=reversible)
        self.to_logits = nn.Linear(dim, num_tokens)
        self.loss_fct = torch.nn.functional.cross_entropy

        self._reset_parameters()

    def forward(self, x, labels=None):
        model_logger.debug(str(x.shape)+str(x.dtype))
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        if labels is not None:
            masked_lm_loss = self.loss_fct(out.view(-1, out.size(-1)), labels.view(-1))
            model_logger.debug(f'loss = {masked_lm_loss}')
            return {'loss': masked_lm_loss, 'logits': out}
        else:
            return {'logits': out}

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        model_logger.info('Sucessful init weight with xavier')
