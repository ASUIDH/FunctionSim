# -*- coding: UTF-8 -*-
import torch
import torch.utils.data as tud


class EmbedDataset(tud.Dataset):
    def __init__(self, trainsamples, samples):
        self.samples = samples
        self.trainsamples = trainsamples

    def __len__(self):
        return len(self.trainsamples)

    def getdata(self, sample):
        embs = sample[0]
        neibors = sample[1]
        embs_lib = sample[2]
        nei_lib = sample[3]
        ind = neibors._indices()
        values = neibors._values()
        ind_lib = nei_lib._indices()
        val_lib = nei_lib._values()
        return ind, values, embs,ind_lib,val_lib,embs_lib

    def collate_one(self, batch):
        maxlen = 0
        maxtup = 0
        maxlibLen = 0
        maxlibTup = 0
        wordsize = batch[0][2].shape[1]
        for sample in batch:
            if (sample[2].shape[0] > maxlen):
                maxlen = sample[2].shape[0]
            if (sample[0].shape[1] > maxtup):
                maxtup = sample[0].shape[1]
            # 以下增加对lib数据的处理
            if (sample[5].shape[0] > maxlibLen):
                maxlibLen = sample[5].shape[0]
            if(sample[3].shape[1] >maxlibTup):
                maxlibTup = sample[3].shape[1]
        newind = torch.zeros([len(batch), 2, maxtup], dtype=torch.int64)
        newvalues = torch.zeros([len(batch), maxtup], dtype=torch.float32)
        newembs = torch.zeros([len(batch), maxlen, wordsize], dtype=torch.float32)
        masks = torch.zeros([len(batch), maxlen])
        #增加对lib数据的处理
        newlibind = torch.zeros([len(batch), 2, maxlibTup], dtype=torch.int64)
        newlibvalues = torch.zeros([len(batch), maxlibTup], dtype=torch.float32)
        newlibembs = torch.zeros([len(batch), maxlibLen, wordsize], dtype=torch.float32)

        for i in range(0, len(batch)):
            sample = batch[i]
            ind = sample[0]
            values = sample[1]
            embs = sample[2]
            #lib
            ind_lib = sample[3]
            value_lib = sample[4]
            embs_lib = sample[5]
            newind[i, :, 0:ind.shape[1]] = ind
            newvalues[i, 0:values.shape[0]] = values
            newembs[i, 0:embs.shape[0], :] = embs
            masks[i, 0:embs.shape[0]] = torch.ones([embs.shape[0]])
            #lib
            newlibind[i,:,0:ind_lib.shape[1]] = ind_lib
            newlibvalues[i,0:value_lib.shape[0]] = value_lib
            newlibembs[i,0:embs_lib.shape[0],:] = embs_lib
        return newind, newvalues, newembs, masks ,newlibind,newlibvalues,newlibembs

    def collate_fn(self, batch):
        samples_batch1 = [batch[i][0] for i in range(len(batch))]
        samples_batch2 = [batch[i][1] for i in range(len(batch))]
        samples_batch3 = [batch[i][2] for i in range(len(batch))]
        return self.collate_one(samples_batch1), self.collate_one(samples_batch2), self.collate_one(samples_batch3)

    def __getitem__(self, idx):
        no1 = self.trainsamples[idx][0]
        no2 = self.trainsamples[idx][1]
        no3 = self.trainsamples[idx][2]
        sample1 = self.samples[no1]
        sample2 = self.samples[no2]
        sample3 = self.samples[no3]
        ind1, values1, embs1,ind_lib1,val_lib1,embs_lib1 = self.getdata(sample1)
        ind2, values2, embs2,ind_lib2,val_lib2,embs_lib2 = self.getdata(sample2)
        ind3, values3, embs3,ind_lib3,val_lib3,embs_lib3 = self.getdata(sample3)
        return [[ind1, values1, embs1,ind_lib1,val_lib1,embs_lib1 ], [ind2, values2, embs2,ind_lib2,val_lib2,embs_lib2],
                [ind3, values3, embs3,ind_lib3,val_lib3,embs_lib3]]  # 后续可能需要补充mask矩阵，长度向量等(因为不需要求平均，所以长度向量暂时没用)
