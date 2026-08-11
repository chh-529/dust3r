"""
Vendored building blocks from DeepSC.

Source chain:
    https://github.com/13274086/DeepSC  (author: HQ Xie)
    -> references/semcom/paper_src/model/architecture/deepsc.py

WHAT WAS COPIED AND WHY
-----------------------
Copied verbatim (modulo the notes below):
    ChannelEncoder, ChannelDecoder                  <- the parts we actually use
    PositionalEncoding, MultiHeadedAttention,
    PositionwiseFeedForward, EncoderLayer,
    DecoderLayer                                    <- for an optional extra semantic
                                                       stage, and for reference

Deliberately NOT copied, because they are natural-language specific and have no
counterpart here:
    Encoder / Decoder     - they start with nn.Embedding(vocab_size, d_model); our
                            input is already a continuous feature, there is no vocab.
                            DeepSC's Decoder is also autoregressive, while DUSt3R
                            produces all patch tokens in parallel.
    create_masks / subsequent_mask
                          - padding and look-ahead masks only make sense for
                            variable-length token sequences. Our token count is fixed.
    DeepSCTokenizer, Metric, loss_function
                          - tokenization, BLEU / sentence similarity, cross entropy.

CHANGES FROM THE ORIGINAL (kept minimal, all listed here)
---------------------------------------------------------
1. `from ...utils import str_type` dropped (was unused by these classes).
2. Type hints and a few docstrings added. No behaviour change.
3. Original quirks are PRESERVED even where they look like bugs, so that this file
   stays a faithful reference; they are flagged with "DeepSC quirk:" comments.
   If you want them fixed, fix them in semcom/codec.py by not using these classes,
   rather than by editing this file.
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:33:53 2020

@author: HQ Xie
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        if mask is not None:
            # fill the masked positions with -1e9
            scores += (mask * -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # DeepSC quirk: `dropout` is ignored here, 0.1 is hardcoded into the sublayers.
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        # post-norm residual (the original Transformer arrangement), not pre-norm
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        # DeepSC quirk: same as EncoderLayer, `dropout` is ignored.
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # q, k, v
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class ChannelEncoder(nn.Module):
    """
    DeepSC's channel encoder: the module that turns a semantic feature into the values
    that are actually put on the air.

    In DeepSC this is Linear(d_model=128 -> 256) -> ReLU -> Linear(256 -> 16), applied
    independently to every token, i.e. each of the 16 outputs per token becomes 8
    complex symbols after real->complex packing.

    Note there is NO activation on the output layer: the transmitted values must be
    free to take any real value (they are amplitudes), and the power constraint is
    imposed afterwards by power normalization, not by a squashing nonlinearity.
    """

    def __init__(self, in_features: int, middle_features: int, out_features: int):
        super(ChannelEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, middle_features),
            nn.ReLU(inplace=True),
            nn.Linear(middle_features, out_features)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ChannelDecoder(nn.Module):
    """
    DeepSC's channel decoder: maps the noisy received values back to the semantic
    feature space.

    In DeepSC this is called as ChannelDecoder(in_features=16, size1=d_model, size2=512):

        x1 = Linear(in_features -> size1)(x)
        x5 = Linear(size2 -> size1)(ReLU(Linear(size1 -> size2)(ReLU(x1))))
        out = LayerNorm(x1 + x5)

    So it is deeper than the encoder and has a residual connection around the inner
    bottleneck. That asymmetry is deliberate and is worth keeping: the decoder has the
    harder job (undoing noise), while the encoder must stay cheap because it runs on
    the transmitter.

    DeepSC quirk: `size2` (512) is LARGER than `size1` (d_model=128), so the "inner"
    part is an expansion rather than a bottleneck, despite the naming.
    """

    def __init__(self, in_features: int, size1: int, size2: int):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output
