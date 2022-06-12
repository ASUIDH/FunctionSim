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
        ind = neibors._indices()
        values = neibors._values()
        shape = neibors.shape
        return ind, values, embs

    def collate_one(self, batch):
        maxlen = 0
        maxtup = 0
        wordsize = batch[0][2].shape[1]
        for sample in batch:
            if (sample[2].shape[0] > maxlen):
                maxlen = sample[2].shape[0]
            if (sample[0].shape[1] > maxtup):
                maxtup = sample[0].shape[1]
        newind = torch.zeros([len(batch), 2, maxtup], dtype=torch.int64)
        newvalues = torch.zeros([len(batch), maxtup], dtype=torch.float32)
        newembs = torch.zeros([len(batch), maxlen, wordsize], dtype=torch.float32)
        masks = torch.zeros([len(batch), maxlen])
        for i in range(0, len(batch)):
            sample = batch[i]
            ind = sample[0]
            values = sample[1]
            embs = sample[2]
            newind[i, :, 0:ind.shape[1]] = ind
            newvalues[i, 0:values.shape[0]] = values
            newembs[i, 0:embs.shape[0], :] = embs
            masks[i, 0:embs.shape[0]] = torch.ones([embs.shape[0]])
        return newind, newvalues, newembs, masks

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
        ind1, values1, embs1 = self.getdata(sample1)
        ind2, values2, embs2 = self.getdata(sample2)
        ind3, values3, embs3 = self.getdata(sample3)
        return [[ind1, values1, embs1], [ind2, values2, embs2],
                [ind3, values3, embs3]]  # 后续可能需要补充mask矩阵，长度向量等(因为不需要求平均，所以长度向量暂时没用)
