# -*- coding: UTF-8 -*-
#针对整个样本的控制流图
import shelve

import os


import torch
import torch.utils.data as tud

from Models.EmbedDatasetWithLib import EmbedDataset
from Models.forwardModelsWithLib import Aggregators, Encoders, Embed
from PrePare.PreData import getGraphsWithLib, getdata,getdataLib


def trainModelsWithVal(index , suffix ,batchsize = 10, word_size = 7 ,embed_size = 32, lr = 3e-3 , cuda = True,epoch = 50,itera =2,depth=2,modelFilePath = "",year = ""):
    bathPath = "/home/yifei/data/wyfdata/pdata/fcg"
    sampleYear = "all"
    splitName = "fcg_{}_datasplit".format(sampleYear)
    fcgName = "fcg_{}_lib".format(sampleYear)
    sampleIdName = "fcg_{}_sample_id".format(sampleYear)
    splitPath = os.path.join(bathPath, splitName)
    sampleIdPath = os.path.join(bathPath, sampleIdName)
    fcgPath = os.path.join(bathPath, fcgName)
    file = shelve.open(os.path.join(sampleIdPath,"sample_id"))
    sample_id = file["sample_id"]
    file.close()
    samples = getGraphsWithLib(fcgPath, sample_id)


    trainsamples = []
    strindex = str(index)
    strsuffix = str(suffix)
    with open(splitPath +"/vt.splits/train." + strsuffix + "/" + "train_mal_" + strindex) as f:
        for line in f.readlines():
            arr = line.strip().split("\t")
            trainsamples.append([int(node) for node in arr])

    dataset = EmbedDataset(trainsamples, samples)
    dataloader = tud.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4,
                                collate_fn=dataset.collate_fn)  # 注意batehsize > 1 时， 图的大小不一样，我也没设置mask矩阵，会报错
    # 注意：稀疏矩阵不支持, num_workers=4 多线程，所以我直接去掉了 pytorch建议在dataloader处先得到 ind，和value load完了再处理成稀疏格式

    if cuda:
        agg = Aggregators(embsize=embed_size, CUDA=cuda).cuda()
        enc = Encoders(word_size=word_size, embed_size=embed_size,depth = depth,cuda = True).cuda()
        model = Embed(embsize=embed_size, word_size=word_size, enc=enc, agg=agg,iterations=itera).cuda()
    else:
        pass
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=lr)

    file = shelve.open(modelFilePath)
    #print("开始训练")
    optCount = 0
    for e in range(epoch):
        for i, (query, target1, target2) in enumerate(dataloader):
            optimizer.zero_grad()
            neibors1, embs1, mask1,nerlib1,embslib1 = getdataLib(query)
            neibors2, embs2, mask2,nerlib2,embslib2 = getdataLib(target1)
            neibors3, embs3, mask3,nerlib3,embslib3 = getdataLib(target2)
            if cuda:
                neibors1 = neibors1.cuda()
                neibors2 = neibors2.cuda()
                neibors3 = neibors3.cuda()
                embs1 = embs1.cuda()
                embs2 = embs2.cuda()
                embs3 = embs3.cuda()
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()
                mask3 = mask3.cuda()
                nerlib1 = nerlib1.cuda()
                nerlib2 = nerlib2.cuda()
                nerlib3 = nerlib3.cuda()
                embslib1 = embslib1.cuda()
                embslib2 = embslib2.cuda()
                embslib3 = embslib3.cuda()

            sample_embeds1 = model(embs1, neibors1,embslib1,nerlib1, mask1)
            sample_embeds2 = model(embs2, neibors2,embslib2,nerlib2, mask2)
            sample_embeds3 = model(embs3, neibors3,embslib3,nerlib3, mask3)  # batch *dim
            score1 = torch.cosine_similarity(sample_embeds1, sample_embeds2, 1)
            score2 = torch.cosine_similarity(sample_embeds1, sample_embeds3, 1)
            loss = -torch.log(torch.sigmoid((score1 - score2))).sum()
            #loss = torch.sum((score1 - 1) * (score1 - 1)) + torch.sum((score2 + 1) *(score2 +1))
            if (optCount % 50 == 0):
                #print("epoch =" + str(e) + " i =" + str(i) + " loss= " + "update time = " + str(optCount) + "  " + str(
                    #loss.item()))
                file["model"] = model
            loss.backward()
            optimizer.step()
            """
            if optCount % 50 == 0 :
                test_file = os.path.join("/home/yifei/data/wyfdata/pdata/fcg/fcg_2019_datasplit/vt.splits/test",
                                         "test_mal_" + str(index)+"val")
                ideal_file = os.path.join("/home/yifei/data/wyfdata/pdata/fcg/fcg_2019_datasplit/vt.splits/ideal",
                                          "ideal_mal_" + str(index)+"val")
                test(cuda = cuda, samples = samples,wordsize = word_size,testBatchsize = 10,modelFilePath = modelFilePath  ,test_file = test_file,ideal_file = ideal_file,testModel = False)
            """
            optCount += 1
    #print("训练完成")
    file.close()

if __name__ == "__main__":
    trainModelsWithVal(index = 1240, suffix =1000,batchsize = 10,word_size = 100 ,embed_size = 128, lr = 3e-3 , cuda = True,epoch = 70)