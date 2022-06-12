# -*- coding: UTF-8 -*-
import os
import shelve

import torch

from Models.TestModels  import TestBatchEmbed,TestBatchEmbedWithLib
from Utils.ComputeMapandNdcg import get_MAP, get_MnDCG, mapSortByValueDESC


def test(cuda, samples,wordsize,testBatchsize,modelFilePath,test_file,ideal_file ,testModel):
    file = shelve.open(modelFilePath)
    model = file["model"]
    test_map = {}
    testmap2 = {}
    line_count = 0
    pedict = TestBatchEmbedWithLib(model, samples, cuda,wordsize,testBatchsize)
    with open(test_file)as f:
        for l in f:
            with torch.no_grad():
                arr = l.strip().split()
                query = int(arr[0])
                map = {}
                candidates = arr[1:]
                candidates = [int(q) for q in candidates]
                value = [v.cpu() for v in pedict(query,candidates)]
                for i in range(0, len(candidates)):
                    map[candidates[i]] = value[i]
                tops_in_line = mapSortByValueDESC(map, 10)  # 返回一个节点的列表，节点按相似度分数大小排序，排前top个点，若长度<top，有多少排多少
                test_map[
                    line_count] = tops_in_line
                testmap2[query] = tops_in_line
                line_count += 1

        line_count = 0
        ideal_map = {}
        ideal_map2 = {}

        with open(ideal_file) as f:
            for l in f:
                arr = l.strip().split()
                arr = [int(x) for x in arr]
                ideal_map[line_count] = arr[1:]
                ideal_map2[arr[0]] = arr[1:]
                line_count += 1
        # Ftools.createDictCSV(testDict=testmap2,idealDict=ideal_map2,fileName=main_dir+"/result_",main_dir=main_dir)
        MAP = get_MAP(10, ideal_map2, testmap2, "/home/yifei/data/wyfdata/prox2/vt_muti")
        MnDCG = get_MnDCG(10, ideal_map2, testmap2)
        if testModel:
            print(test_file + "计算完成")
        else :
            print("val 计算完成")
        #print(MAP)
        #print(MnDCG)
        return MAP,MnDCG
if __name__ == "__main__":
    """
    file = shelve.open("/home/yifei/data/wyfdata/pdata/cfg-sample-info/cfg-sample-id")
    sample_id = file["sample_id"]
    file.close()
    base_dir = "/home/yifei/data/wyfdata/pdata/ana_cfg/"
    samples = getGraphs(base_dir, sample_id)
    index = 1240
    test(cuda = True,index = 1240, suffix = 1000,samples =samples)
    """