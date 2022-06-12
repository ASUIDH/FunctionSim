# -*- coding: UTF-8 -*-
import os
import shelve

import torch


def getGraphs(base_dir,sample_id):
    samples = [0] * len(sample_id)
    for sample_path in os.listdir(base_dir):
        file = shelve.open(base_dir + sample_path +"/" + "cfgs")
        methods = file["cfgs"]
        file.close()
        filename = sample_path.strip().split("_")[0]
        samples[sample_id[filename]] = methods
    print( "样本加载完成")
    return samples
def getGraphsWithKey(base_dir,sample_id,key):
    samples = [0] * len(sample_id)
    for sample_path in os.listdir(base_dir):
        if not sample_path.endswith("dir"):
            continue
        filename = sample_path.strip().split(".")[0]
        file = shelve.open(os.path.join(base_dir,filename))
        methods = file[key]
        file.close()
        samples[sample_id[filename]] = methods
    print( "样本加载完成")
    return samples
def getGraphsWithLib(base_dir,sample_id):
    samples = [0] * len(sample_id)
    for sample_path in os.listdir(base_dir):
        if not sample_path.endswith("dir"):
            continue
        filename = sample_path.strip().split(".")[0]
        file = shelve.open(os.path.join(base_dir, filename))
        methods = file["fcg"]
        methodsL = file["fcgL"]
        file.close()
        samples[sample_id[filename]] = methods+methodsL
    print("样本加载完成")
    return samples
def getGraphsForMatch(base_dir,sample_id):
    samples = [0] * len(sample_id)
    for sample_path in os.listdir(base_dir):
        if not sample_path.endswith("dir"):
            continue
        filename = sample_path.strip().split(".")[0]
        file = shelve.open(os.path.join(base_dir, filename))
        methods = file["fcg"]
        file.close()
        samples[sample_id[filename]] = methods
    print("样本加载完成")
    return samples
def getGraphsWithOutLib(base_dir,sample_id):
    #不带上lib函数就好了
    samples = [0] * len(sample_id)
    for sample_path in os.listdir(base_dir):
        if not sample_path.endswith("dir"):
            continue
        filename = sample_path.strip().split(".")[0]
        file = shelve.open(os.path.join(base_dir, filename))
        methods = file["fcg"]
        file.close()
        samples[sample_id[filename]] = methods
    print("样本加载完成")
    return samples
def getdataLib(query):
    ind = query[0]
    values = query[1]
    embs = query[2]
    mask = query[3]
    ind_lib = query[4]
    values_lib = query[5]
    embslib = query[6]
    shape = embs.shape[1]
    shape_lib = embslib.shape[1]
    ner = torch.stack([torch.sparse.FloatTensor(ind[j],values[j],torch.Size([shape,shape])).coalesce() for j in range(ind.shape[0])])
    nerlib = torch.stack([torch.sparse.FloatTensor(ind_lib[j],values_lib[j],torch.Size([shape,shape_lib])).coalesce() for j in range(ind_lib.shape[0])])
    return ner , embs,mask,nerlib,embslib

def getdata(query):
    ind = query[0]
    values = query[1]
    embs = query[2]
    mask = query[3]
    shape = embs.shape[1]
    ner = torch.stack([torch.sparse.FloatTensor(ind[j],values[j],torch.Size([shape,shape])).coalesce() for j in range(ind.shape[0])])
    return ner , embs,mask

if __name__ == "__main__":
    file = shelve.open("/home/yifei/data/wyfdata/pdata/fcg/fcg_2019_sample_id/sample_id")
    sample_id = file["sample_id"]
    a = getGraphsWithLib("/home/yifei/data/wyfdata/pdata/fcg/fcg_2019_doc2vec_lib",sample_id)
    print(a)