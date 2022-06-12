# -*- coding: UTF-8 -*-
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import shelve

from PrePare.PreData import  getGraphsWithLib
from TrainForFcg.train_embed_lib import trainModelsWithVal
from TestForFcg.test_embed_lib import test
from Utils.setup_seed import setup_seed


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bathPath = "/home/yifei/data/wyfdata/pdata/fcg"
    sampleYear = "all"
    splitName = "fcg_{}_datasplit".format(sampleYear)
    fcgName = "fcg_{}_lib".format(sampleYear)
    sampleIdName = "fcg_{}_sample_id".format(sampleYear)
    splitPath = os.path.join(bathPath,splitName)
    sampleIdPath = os.path.join(bathPath,sampleIdName)
    fcgPath = os.path.join(bathPath,fcgName)
    seed = 3015
    setup_seed(seed)
    indexs = [1238]
    suffixs = [1000]
    epochs = [25]
    for index in indexs:
        for i in range (len(suffixs)):
            suffix = suffixs[i]
            epoch = epochs[i]
            modelFilePath = os.path.join(
                splitPath,"vt.trainModels/train." + str(suffix),
                str(index) + "embed_lib")
            print("当前参数为： " + "index : " +str(index) + "epoch : " +str(epoch) + "suffix : " + str(suffix))
            trainModelsWithVal(index = index, suffix =suffix,batchsize = 64,word_size = 8 ,embed_size = 64, lr = 1e-4 , cuda = True,epoch = epoch,itera = 2,depth = 2,modelFilePath=modelFilePath,year = "all")
            file = shelve.open(os.path.join(sampleIdPath,"sample_id"))
            sample_id = file["sample_id"]
            file.close()
            samples = getGraphsWithLib(fcgPath, sample_id)

            test_file = os.path.join(splitPath,"vt.splits/test",
                                     "test_mal_" + str(index))
            ideal_file = os.path.join(splitPath,"vt.splits/ideal",
                                      "ideal_mal_" + str(index))
            map,ndcg = test(cuda=True,samples=samples,wordsize = 8,testBatchsize = 50,modelFilePath = modelFilePath,test_file=test_file,ideal_file=ideal_file ,testModel=True)
            print(map)
            print(ndcg)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
