# -*- coding: UTF-8 -*-
import torch
from torch import nn



class Test(nn.Module):
    def __init__(self, model,samples,cuda):
        ''' 初始化输关系和实体embedding
        '''
        super(Test, self).__init__()
        self.model = model
        self.samples = samples
        self.cuda =cuda

    def forward(self, query, target):
        if self.cuda:
            embs1 = self.samples[query][0].unsqueeze(0).cuda()
            neibors1 = self.samples[query][1].unsqueeze(0).cuda()
            mask1 = torch.ones([1,embs1.shape[1]]).cuda()
            embs2 = self.samples[target][0].unsqueeze(0).cuda()
            neibors2 = self.samples[target][1].unsqueeze(0).cuda()
            mask2 = torch.ones([1,embs2.shape[1]]).cuda()
        else:
            embs1 = self.samples[query][0].unsqueeze(0)
            neibors1 = self.samples[query][1].unsqueeze(0)
            mask1 = torch.ones([1,embs1.shape[1]])
            embs2 = self.samples[target][0].unsqueeze(0)
            neibors2 = self.samples[target][1].unsqueeze(0)
            mask2 = torch.ones([1,embs2.shape[1]])
        sample_embeds1 = self.model(embs1,neibors1,mask1)
        sample_embeds2 = self.model(embs2,neibors2,mask2)
        score = torch.cosine_similarity(sample_embeds1,sample_embeds2,1)
        return score
class TestBatchEmbed(nn.Module):
    def __init__(self, model,samples,cuda,wordsize,testBatchsize):
        ''' 初始化输关系和实体embedding
        '''
        super(TestBatchEmbed, self).__init__()
        self.model = model
        self.samples = samples
        self.cuda = cuda
        self.embeds = {} #测试前，先从这里面找，如果有就不用重复测试了；如果没有再重复测试。
        self.wordsize = wordsize
        self.testBatchsize = testBatchsize
    def forward(self, query, targets):
        nowInd = 0
        scores = [-1000] * len(targets) #先初始化好长度

        #先找到query_emb
        query_emb = None
        if query in self.embeds:
            query_emb = self.embeds[query]
        else:
            if self.cuda:
                embs1 = self.samples[query][0].unsqueeze(0).cuda()
                neibors1 = self.samples[query][1].unsqueeze(0).cuda()
                mask1 = torch.ones([1, embs1.shape[1]]).cuda()
            else:
                embs1 = self.samples[query][0].unsqueeze(0)
                neibors1 = self.samples[query][1].unsqueeze(0)
                mask1 = torch.ones([1, embs1.shape[1]])
            query_emb = self.model(embs1, neibors1, mask1)[0] #本来是一个 1* embsize 的tensor
            self.embeds[query] = query_emb

        while nowInd < len(targets):
            nowTargets = [] #target用来确定id
            nowInds = [] # inds用来确定score中的位置
            while len(nowTargets) < self.testBatchsize and nowInd < len(targets):
                nowTarget = targets[nowInd]
                if nowTarget in self.embeds:
                    nowScore = torch.cosine_similarity(query_emb,self.embeds[nowTarget],0)
                    scores[nowInd] = nowScore #只改当前位置的值
                else :
                    nowTargets.append(nowTarget)
                    nowInds.append(nowInd)
                nowInd += 1
            # 两种情况，找到10个了，开始测试，或者不到10个，但是nowInd超了，都把nowTarget中剩下的样本测试了，然后开始下一轮循环。

            if len(nowTargets) == 0 : #这次没有 实际上就是结束了
                continue

            maxLen = 0
            maxTup = 0
            embs = []
            inds = []
            values = []

            # 初始化一下targetNode的信息
            for targetId in nowTargets:

                emb = self.samples[targetId][0]
                neibors = self.samples[targetId][1]
                ind = neibors._indices()
                value = neibors._values()

                embs.append(emb)
                inds.append(ind)
                values.append(value)

            #统计出最多的边和最多的结点
            for i in range(len(nowTargets)):
                nowLen = embs[i].shape[0]
                nowTup = inds[i].shape[1]
                if nowLen > maxLen:
                    maxLen = nowLen
                if nowTup > maxTup:
                    maxTup = nowTup

            newind = torch.zeros([len(nowTargets), 2, maxTup], dtype=torch.int64)
            newvalues = torch.zeros([len(nowTargets), maxTup], dtype=torch.float32)
            newembs = torch.zeros([len(nowTargets), maxLen, self.wordsize], dtype=torch.float32)
            masks = torch.zeros([len(nowTargets) , maxLen])

            #更新模型输入
            for i in range(len(nowTargets)):

                ind = inds[i]
                value = values[i]
                emb = embs[i]

                # 更新目前的输入
                newind[i,:,0:ind.shape[1]] = ind
                newvalues[i, 0:value.shape[0]] = value
                newembs[i, 0:emb.shape[0], :] = emb
                masks[i, 0:emb.shape[0]] = torch.ones([emb.shape[0]])
            ner = torch.stack(
                [torch.sparse.FloatTensor(newind[j], newvalues[j], torch.Size([maxLen, maxLen])).coalesce() for j in
                 range(newind.shape[0])])
            #转移到cuda
            if self.cuda:
                newembs = newembs.cuda()
                ner = ner.cuda()
                masks = masks.cuda()
            sample_embeds = self.model(newembs, ner, masks)

            #根据sample_embeds，和targetId去更新score

            for i in range(len(nowTargets)):
                target = nowTargets[i]
                nowInd = nowInds[i]  # 在target数组中的位置
                emb = sample_embeds[i]
                #更新保存的embeds
                self.embeds[target] = emb
                nowScore = torch.cosine_similarity(query_emb, self.embeds[target],0)
                scores[nowInd] = nowScore
        return scores
class TestBatchEmbedWithLib(nn.Module):
    def __init__(self, model,samples,cuda,wordsize,testBatchsize):
        ''' 初始化输关系和实体embedding
        '''
        super(TestBatchEmbedWithLib, self).__init__()
        self.model = model
        self.samples = samples
        self.cuda = cuda
        self.embeds = {} #测试前，先从这里面找，如果有就不用重复测试了；如果没有再重复测试。
        self.wordsize = wordsize
        self.testBatchsize = testBatchsize
    def forward(self, query, targets):
        nowInd = 0
        scores = [-1000] * len(targets) #先初始化好长度

        #先找到query_emb
        query_emb = None
        if query in self.embeds:
            query_emb = self.embeds[query]
        else:
            if self.cuda:
                embs1 = self.samples[query][0].unsqueeze(0).cuda()
                neibors1 = self.samples[query][1].unsqueeze(0).cuda()
                embs_lib = self.samples[query][2].unsqueeze(0).cuda()
                nei_lib = self.samples[query][3].unsqueeze(0).cuda()
                mask1 = torch.ones([1, embs1.shape[1]]).cuda()
            else:
                embs1 = self.samples[query][0].unsqueeze(0)
                neibors1 = self.samples[query][1].unsqueeze(0)
                embs_lib = self.samples[query][2].unsqueeze(0)
                nei_lib = self.samples[query][3].unsqueeze(0)
                mask1 = torch.ones([1, embs1.shape[1]])
            query_emb = self.model(embs1, neibors1,embs_lib,nei_lib, mask1)[0] #本来是一个 1* embsize 的tensor
            self.embeds[query] = query_emb
        while nowInd < len(targets):
            nowTargets = [] #target用来确定id
            nowInds = [] # inds用来确定score中的位置
            while len(nowTargets) < self.testBatchsize and nowInd < len(targets):
                nowTarget = targets[nowInd]
                if nowTarget in self.embeds:
                    nowScore = torch.cosine_similarity(query_emb,self.embeds[nowTarget],0)
                    scores[nowInd] = nowScore #只改当前位置的值
                else :
                    nowTargets.append(nowTarget)
                    nowInds.append(nowInd)
                nowInd += 1
            # 两种情况，找到10个了，开始测试，或者不到10个，但是nowInd超了，都把nowTarget中剩下的样本测试了，然后开始下一轮循环。

            if len(nowTargets) == 0 : #这次没有 实际上就是结束了
                continue

            maxLen = 0
            maxTup = 0
            maxLen_lib = 0
            maxTup_lib = 0
            embs = []
            inds = []
            values = []
            #增加lib信息
            embs_lib=[]
            inds_lib =[]
            values_lib=[]
            # 初始化一下targetNode的信息
            for targetId in nowTargets:

                emb = self.samples[targetId][0]
                neibors = self.samples[targetId][1]
                emb_lib = self.samples[targetId][2].unsqueeze(0).cuda()
                nei_lib = self.samples[targetId][3].unsqueeze(0).cuda()
                ind = neibors._indices()
                value = neibors._values()
                #lib
                ind_lib = nei_lib._indices()
                value_lib = nei_lib._values()
                embs.append(emb)
                inds.append(ind)
                values.append(value)
                #lib
                embs_lib.append(emb_lib)
                inds_lib.append(ind_lib)
                values_lib.append(value_lib)

            #统计出最多的边和最多的结点
            for i in range(len(nowTargets)):
                nowLen = embs[i].shape[0]
                nowTup = inds[i].shape[1]
                nowLen_lib = embs_lib[i].shape[0]
                nowTup_lib = inds_lib[i].shape[1]
                if nowLen > maxLen:
                    maxLen = nowLen
                if nowTup > maxTup:
                    maxTup = nowTup
                # lib
                if nowLen_lib > maxLen_lib:
                    maxLen_lib = nowLen_lib
                if nowTup_lib > maxTup_lib:
                    maxTup_lib = nowTup_lib

            newind = torch.zeros([len(nowTargets), 2, maxTup], dtype=torch.int64)
            newvalues = torch.zeros([len(nowTargets), maxTup], dtype=torch.float32)
            newembs = torch.zeros([len(nowTargets), maxLen, self.wordsize], dtype=torch.float32)
            masks = torch.zeros([len(nowTargets) , maxLen])
            #lib
            newind_lib = torch.zeros([len(nowTargets), 2, maxTup_lib], dtype=torch.int64)
            newvalues_lib = torch.zeros([len(nowTargets), maxTup_lib], dtype=torch.float32)
            newembs_lib = torch.zeros([len(nowTargets), maxLen_lib, self.wordsize], dtype=torch.float32)

            #更新模型输入
            for i in range(len(nowTargets)):

                ind = inds[i]
                value = values[i]
                emb = embs[i]

                # 更新目前的输入
                newind[i,:,0:ind.shape[1]] = ind
                newvalues[i, 0:value.shape[0]] = value
                newembs[i, 0:emb.shape[0], :] = emb
                masks[i, 0:emb.shape[0]] = torch.ones([emb.shape[0]])
            ner = torch.stack(
                [torch.sparse.FloatTensor(newind[j], newvalues[j], torch.Size([maxLen, maxLen])).coalesce() for j in
                 range(newind.shape[0])])
            #lib
            ner_lib = torch.stack([torch.sparse.FloatTensor(newind_lib[j],newvalues_lib[j],torch.Size([maxLen,maxLen_lib])).coalesce() for j in range(newind_lib.shape[0])])
            #转移到cuda
            if self.cuda:
                newembs = newembs.cuda()
                ner = ner.cuda()
                masks = masks.cuda()
            sample_embeds = self.model(newembs, ner,newembs_lib,ner_lib, masks)

            #根据sample_embeds，和targetId去更新score

            for i in range(len(nowTargets)):
                target = nowTargets[i]
                nowInd = nowInds[i]  # 在target数组中的位置
                emb = sample_embeds[i]
                #更新保存的embeds
                self.embeds[target] = emb
                nowScore = torch.cosine_similarity(query_emb, self.embeds[target],0)
                scores[nowInd] = nowScore
        return scores