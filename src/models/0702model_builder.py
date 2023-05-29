import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder, Differ, SentEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('/home/NLP/pre-training-model/bert-base')
        # 是否保留梯度
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        # todo 设置合适的句子数量上限
        self.dim_attention = SentEncoder(args.ext_heads, self.bert.model.config.hidden_size, args.ext_dropout, args.dim_layers)
        self.differ = Differ(hidden_size=768, max_sents_num=10, differ_layer_num=4)
        self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        if args.encoder == 'baseline':
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads,
                                     intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        # todo 把句子们分开投入DimAtention
        sents_vec = None
        for corpus, attmask, flag in zip(top_vec, mask_src, clss):
            # 针对单一条的语料，先把embeddings开来，以句子为单位
            # 最终期望得到的sents:[m1,m2,d_model]，
            # m1是该条语料包含的句子的个数，m2是该条语料中最大的句子长度
            # corpus:[max_seq_len, d_model]
            a = torch.nonzero(attmask == False)
            # 获取整句的句长
            max_sent_len = int(a[0]) if a.shape[0] != 0 else int(top_vec.shape[1])
            sent_num = flag.shape[-1]
            # 第一次出现0的位置应当存放最大的标记，标记值为max_sent_len
            zero_index = torch.nonzero(flag == 0)

            max_flag = zero_index[1] if zero_index.shape[0] > 1 else 0
            if flag[-1] == 0:
                if max_flag != 0:
                    flag[max_flag] = max_sent_len
            else:
                # 如果最后一个位置不是0，则说明句子数量是够的，
                # 这时候只需要找到对应的实际句子长度塞进去就好
                b = torch.tensor(max_sent_len, device=flag.device).unsqueeze(0).unsqueeze(0)
                flag = torch.unsqueeze(flag, dim=0)
                flag = torch.cat((flag, b), dim=1).squeeze(0).squeeze(0)

            seg = []
            for i in range(flag.shape[-1] - 1):
                if flag[i + 1] - flag[i] > 0:
                    seg.append(int(- flag[i] + flag[i + 1]))
                else:
                    break

            seg = tuple(seg)
            corpus = corpus[:max_sent_len]
            # sents是一个tuple，里边每一项对应一个句子的所有词嵌入
            sents = list(torch.split(corpus, seg, dim=0))
            new_sent_vec = None
            for sent_embeddings in sents:
                sent_vec = self.dim_attention(sent_embeddings).unsqueeze(0)
                new_sent_vec = torch.cat((new_sent_vec, sent_vec), dim=0) if new_sent_vec is not None else sent_vec
            # new_sent_vec:[n, 768]
            # 此时需要判断new_sent_vec中包含的句子嵌入的个数n，
            # 如果与mask_cls不一致，则需要补齐。
            # 由于补齐的部分会在下边通过mask直接进行置零不参与运算，这里随便补就好,就随手补一下sent_vec吧
            if new_sent_vec.shape[0] < mask_cls.shape[1]:
                for i in range(mask_cls.shape[1] - new_sent_vec.shape[0]):
                    new_sent_vec = torch.cat((new_sent_vec, sent_vec), dim=0)
            new_sent_vec = new_sent_vec.unsqueeze(0)
            sents_vec = torch.cat((sents_vec, new_sent_vec), dim=0) if sents_vec is not None else new_sent_vec
        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = get_sent_vet(top_vec, mask_src, clss)
        # 对多余的句子padding进行掩码
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        # todo 0.引入DimAttention
        # todo 1.引入差分放大器
        sents_vec = self.differ(sents_vec)
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


def get_sent_vet(top_vec, mask_src, clss):
    batch_vec = None
    for corpus, attmask, flag in zip(top_vec, mask_src, clss):
        sents_vec = None
        for i in range(len(flag) - 1):
            a = flag[i]
            b = flag[i + 1]
            # 判断这条语料是否含有字符padding
            c = torch.nonzero(attmask == False)
            # 有padding的话
            if c.shape[0] != 0:
                max_sent_len = int(c[0])
            # 没有padding的话
            else:
                max_sent_len = corpus.shape[0]
            # 求clss中每个标志位直接的距离，准备去句子嵌入求和。
            # 由于为了对齐可能会含有句子padding，因此需要额外判断b是否一定大于a
            if b - a > 0:
                sent_vec = torch.sum(corpus[a:b, :], dim=0) / (b - a)
            else:
                sent_vec = torch.sum(corpus[a:max_sent_len, :], dim=0) / (max_sent_len - a)
            sent_vec = sent_vec.unsqueeze(0)
            sents_vec = sent_vec if sents_vec is None else torch.cat((sents_vec, sent_vec), dim=0)
            # 再补上句子padding
        if len(sents_vec) < len(flag):
            for _ in range(len(flag) - len(sents_vec)):
                sents_vec = sent_vec if sents_vec is None else torch.cat((sents_vec, sent_vec), dim=0)
        batch_vec = sents_vec.unsqueeze(0) if batch_vec is None else torch.cat((batch_vec, sents_vec.unsqueeze(0)), dim=0)
    return batch_vec


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
