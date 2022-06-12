import numpy

def mapSortByValueDESC(map,top):
    """
    sort by value desc
    """
    if top>len(map):
        top=len(map)
    items=map.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    e=[ backitems[i][1] for i in range(top)]
    return e


def mapSortByValueASC(map,top):
    """
    sort by value asc
    """
    if top>len(map):
        top=len(map)
    items=map.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort()
    e=[ backitems[i][1] for i in range(top)]
    return e


def get_AP(k, ideal, test):
    """
        compute AP
    """
    ideal = set(ideal)
    accumulation = 0.0
    count = 0
    for i in range(len(test)):
        if i >= k:
            break
        if test[i] in ideal:
            count += 1
            accumulation += count / (i + 1.0)
    m = len(ideal)
    n = k
    x = 0
    if m > n:
        x = n
    else:
        x = m
    if x == 0:
        return 0
    return accumulation / x


def get_APandprint(k, ideal, test):
    """
        compute AP
    """
    ideal = set(ideal)
    accumulation = 0.0
    count = 0
    for i in range(len(test)):
        if i >= k:
            break
        if test[i] in ideal:
            count += 1
            accumulation += count / (i + 1.0)
    m = len(ideal)
    n = k
    x = 0
    if m > n:
        x = n
    else:
        x = m
    if x == 0:
        return 0
    if accumulation / x < 0.1:
        pass
    return accumulation / x


def get_MAP(k, ideal_map, test_map, main_dir):
    """
        compute MAP
    """

    accumulation = 0.0
    ideal_bad = {}
    test_bad = {}
    for key in ideal_map.keys():
        ap = get_APandprint(k, ideal_map[key], test_map[key])
        if ap == 0:
            test_bad[key] = test_map[key]
            ideal_bad[key] = ideal_map[key]
        accumulation += ap  # 这里我改成test无可厚非吧？
    # Ftools.createDictCSV(testDict=test_bad, idealDict=ideal_bad, fileName=main_dir+"/relustbad_",main_dir=main_dir)
    if len(ideal_map) == 0:
        return 0

    return accumulation / len(ideal_map)


def get_nDCG(k, ideal, test):
    """
        compute NDCG
    """
    ideal = set(ideal)
    accumulation = 0.0
    for i in range(len(test)):
        if i >= k:
            break
        if test[i] in ideal:
            if i == 0:
                accumulation += 1.0
            else:
                accumulation += 1.0 / numpy.log2(i + 1)
    normalization = 0.0
    for i in range(len(ideal)):
        if i >= k:
            break
        if i == 0:
            normalization += 1.0
        else:
            normalization += 1.0 / numpy.log2(i + 1)
    if normalization == 0:
        return 0
    return accumulation / normalization


def get_MnDCG(k, ideal_map, test_map):
    """
        compute mean NDCG
    """

    accumulation = 0.0
    for key in ideal_map.keys():
        accumulation += get_nDCG(k, ideal_map[key], test_map[key])
    if len(ideal_map) == 0:
        return 0
    return accumulation / len(ideal_map)
