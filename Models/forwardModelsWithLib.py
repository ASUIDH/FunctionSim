# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.nn import init
from Models.SigmodLayer import SigmodLayer

class Encoders(nn.Module):
    def __init__(self,word_size,embed_size,depth,cuda):
        ''' 初始化输关系和实体embedding
        '''
        super(Encoders, self).__init__()
        self.embed_size = embed_size
        self.word_size = word_size
        self.depth =depth
        self.weight = nn.Parameter(torch.FloatTensor(self.word_size,self.embed_size))
        self.sig1 = SigmodLayer(embed_size=self.embed_size, depth=self.depth, cuda=cuda)
        self.sig2 = SigmodLayer(embed_size=self.embed_size, depth=self.depth, cuda=cuda)
        if cuda:
            self.sig1.cuda()
            self.sig2.cuda()
            pass
        init.xavier_uniform(self.weight)

    def forward(self, word_embs, agg_embs_func, agg_embed_lib):
        #word_embs  batch * size *wdim
        #agg_embs batch * *size *edim

        new_word_embs = torch.matmul(word_embs,self.weight)  #batch * size * edim
        new_agg_embs_func = self.sig1(agg_embs_func)
        new_agg_embs_lib = self.sig2(agg_embed_lib) # batch *size *edim
        embs = torch.tanh(new_word_embs + new_agg_embs_func + new_agg_embs_lib) # batch *size *edim
        return embs



class Aggregators(nn.Module):
    def __init__(self,embsize,CUDA):
        super(Aggregators, self).__init__()
        self.embsize = embsize
        self.CUDA = CUDA
    def forward(self,last_embs,neibors):
        #neibors batch *size *size
        #last_embs = batch*size *emb
        #embs 还是 batch * size * edim
        if self.CUDA:
            embs = torch.stack([torch.sparse.mm(neibors[i].cuda(),last_embs[i].cuda())for i in range(neibors.shape[0])])
        else:
            embs = torch.stack([torch.sparse.mm(neibors[i],last_embs[i])for i in range(neibors.shape[0])])
        return embs
class Embed(nn.Module):
    def __init__(self,embsize,word_size,enc,agg,iterations =2):
        super(Embed, self).__init__()
        #edim *edim
        self.iteras = iterations
        self.enc =enc
        self.agg = agg
        self.wordsize =word_size
        self.embsize =embsize
        self.weight = nn.Parameter(torch.FloatTensor(self.wordsize,self.embsize))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.embsize,self.embsize))
        init.xavier_uniform(self.weight)
        init.xavier_uniform(self.weight2)

    def forward(self,word_embs,neibors,lib_embs,neibors_lib,mask):

        last_embed = torch.zeros(list(word_embs.shape[:-1])+[self.embsize],dtype = torch.float32) # batch * size *edim
        for j in range(self.iteras):
            agg_embed_func = self.agg(last_embed,neibors)
            agg_embed_lib = self.agg(lib_embs,neibors_lib)
            agg_embed_lib = torch.matmul(agg_embed_lib,self.weight)
            last_embed = self.enc(word_embs,agg_embed_func,agg_embed_lib)
        last_embed = last_embed  * mask.unsqueeze(2) #给最后计算的每个结点的向量乘，不该有的最后就没了
        all_sample_embs = last_embed.sum(1) #所有结点相加，故sum维度为2
        all_sample_embs = torch.mm(all_sample_embs , self.weight2)
        #这里直接把 不同方法 的结果聚合在一起
        return all_sample_embs