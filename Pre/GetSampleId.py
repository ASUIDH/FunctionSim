import os
import shelve


def getid(base_dir):
    id_sample=[]
    sample_id={}
    for sample_path in os.listdir(base_dir):
        if not sample_path.endswith("dir"):
            continue
        sample = sample_path.strip().split(".")[0]
        id_sample.append(sample)
        sample_id[sample] = len(sample_id)
    return id_sample,sample_id
if __name__ == "__main__":
    baseDir = "/home/yifei/data/wyfdata/pdata/fcg/fcg_all_lib"
    id_sample,sample_id = getid(baseDir)
    file = shelve.open("/home/yifei/data/wyfdata/pdata/fcg/fcg_all_sample_id/sample_id")
    file["id_sample"] = id_sample
    file["sample_id"] = sample_id
    print("end")
    file.close()