import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DimAttentionLayer

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x)
        h = self.linear2(h).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class Differ(nn.Module):
    def __init__(self, hidden_size, max_sents_num, differ_layer_num):
        super(Differ, self).__init__()
        self.max_sents_num = max_sents_num
        self.differ_layer_num = differ_layer_num
        self.differ_layers = nn.ModuleList([DifferLayer(hidden_size, max_sents_num) for _ in range(differ_layer_num)])

    def forward(self, sents_embeddings, mask_cls=None):
        for i in range(self.differ_layer_num):
            sents_embeddings = self.differ_layers[i](sents_embeddings, mask_cls)
        return sents_embeddings


class DifferLayer(nn.Module):
    def __init__(self, hidden_size, max_sents_num):
        super(DifferLayer, self).__init__()
        # 设定一个最大的句子数量
        self.max_sents_num = max_sents_num
        self.amplifiers = nn.ModuleList([Amplifier(hidden_size) for _ in range(max_sents_num)])

    def forward(self, sents_embeddings, mask_cls=None):
        """
        sent_embeddings:[bsz,sents_num,hidden_size]
        mask_cls:[bsz,sents_num]
        """
        sents_num = sents_embeddings.shape[-2]
        new_embeddings = None
        for i in range(sents_num):
            # v_in_0:[bsz, hidden_size]
            v_in_0 = sents_embeddings[:, i, :]
            v_in_1 = torch.sum(sents_embeddings, dim=1) - v_in_0
            v_in_1 = v_in_1 / (sents_num - 1)
            # 当i超过max_sents_num时，调用前边的差分器
            if i < self.max_sents_num:
                res = self.amplifiers[i](v_in_0, v_in_1)
            else:
                a = i
                while a >= self.max_sents_num:
                    a = a - self.max_sents_num
                res = self.amplifiers[a](v_in_0, v_in_1)
            new_embeddings = res.unsqueeze(1) if new_embeddings is None else torch.cat(
                (new_embeddings, res.unsqueeze(1)), dim=1)
        return new_embeddings


# 单独一个差分器
class Amplifier(nn.Module):
    def __init__(self, hidden_size):
        super(Amplifier, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, v_0, v_1):
        # v_0代表v_in+，v_1代表v_in-,v_0:[bsz,hidden_size]
        v_out = self.linear(v_0 - v_1) + v_0
        return v_out

class SentEncoder(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, dim_layer_num=4):
        super(SentEncoder, self).__init__()
        self.dim_attention = nn.ModuleList()
        self.dim_layer_num = dim_layer_num
        self.head_count = head_count
        self.model_dim = model_dim
        self.dropout = dropout
        self.AttList = nn.ModuleList([DimAttentionLayer(self.head_count, self.model_dim, self.dropout)
                                      for _ in range(self.dim_layer_num)])

    def forward(self, input_embeddings):
        input_embeddings = input_embeddings
        for i in range(self.dim_layer_num):
            input_embeddings = self.AttList[i](input_embeddings)
        SentEmbeddings = torch.sum(input_embeddings, dim=-2) / input_embeddings.shape[-2]
        return SentEmbeddings
