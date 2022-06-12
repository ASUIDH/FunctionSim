import csv
import os
import shelve

# do this file with server ,in other project

def createListCSV(fileName="",name=None,dataList=[],delimiter=','):
    with open(fileName, "w", newline="", encoding='utf-8') as csvFile:
        csvWriter = csv.writer(csvFile,delimiter=delimiter)
        if name != None:
            csvWriter.writerow(name)
        for data in dataList:
            if(len(data)==0):
                continue
            csvWriter.writerow(data)
        csvFile.close()
def getTrainandTestDataset(datasplitpath ,dirpath,sample_id):
    for middle_path in os.listdir(datasplitpath):
        newdir = dirpath +"vt.splits/" + middle_path +"/"
        if("train" in middle_path) :
            train_model_dir = dirpath+"vt.trainModels/" + middle_path +"/"
            if not os.path.exists(train_model_dir):
                os.makedirs(train_model_dir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        for file_name in os.listdir(datasplitpath + middle_path):
            print(file_name)
            full_path = datasplitpath + middle_path + "/" +file_name
            with open (full_path) as f :
                newlines=[]
                for line in f.readlines():
                    nowline = []
                    arr = line.strip().split("\t")
                    if arr[0] not in sample_id: #第一个都没就不判断了
                        continue
                    for node in arr:
                        if not node in sample_id and len(arr) >3: #测试集就把这个结点舍弃
                            continue
                        elif not node in sample_id and len(arr) == 3 and "train" in  full_path: #训练集 或者 测试集第一个结点，就将其整组去除掉
                            break
                        elif not node in sample_id and len(arr) < 3 and "train" not in  full_path:
                            break
                        elif not node in sample_id: #其实这个没办法处理，我觉得后面也可能会报错，后续再处理吧，因为没有相似的节点了
                            continue
                        nowline.append(sample_id[node]) #在这还能没这个key？我不都给你过滤了吗？？？
                    if(len(nowline))<3: #训练集后两个出问题了，还是会到这里，在这里去除掉
                        continue
                    newlines.append(nowline)
            createListCSV(fileName = newdir + file_name, dataList = newlines,
                      delimiter='\t')

if __name__ == "__main__":
    file = shelve.open("/home/yifei/data/wyfdata/pdata/fcg/fcg_all_sample_id/sample_id")
    sample_id = file["sample_id"]
    getTrainandTestDataset("/home/yifei/data/wyfdata/pdata/fcg/fcg_all_datasplit_md5/", "/home/yifei/data/wyfdata/pdata/fcg/fcg_all_datasplit/",
                           sample_id=sample_id)