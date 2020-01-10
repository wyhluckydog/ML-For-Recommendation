import numpy as np
from collections import OrderedDict
import functools
def cmp(x, y):
    if x[0] > y[0]:
        return 1
    if x[0] < y[0]:
        return -1
    if x[1] > y[1]:
        return 1
    if x[1] < y[1]:
        return -1
    return 0

def preprocess(filename):
    data = OrderedDict()
    with open(filename,'r') as f:
        line = f.readline()
        line = f.readline()
        while line != None and line != "":
            record = list(line.split(','))
            userid = record[1][1:-1]
            productid = record[5][1:-1]
            rating = record[6][1:-1]
            if (userid, productid) in data.keys():
                data[(userid, productid)] += rating
            else:
                data[(userid, productid)] = rating
            line = f.readline()
    # data = sorted(data, cmp=lambda x, y:cmp((x[0],x[1]), (y[0], y[1])))
    data = OrderedDict(sorted(data.items(), key=functools.cmp_to_key(cmp)))
    with open("Data/tafeng_all_data.rating",'w') as f:
        for (userid, productid) in data.keys():
            # record = '\t'.join((userid, productid, data[(userid, productid)]))
            f.writelines('\t'.join((userid, productid, data[(userid, productid)])) + '\n')

    print("Preprocess Finished!")

def mapUserandItem(filename):
    useridList = []
    itemidList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i, l = arr[0], arr[1], int(arr[2])
            if u not in useridList:
                useridList.append(u)
            if i not in itemidList:
                itemidList.append(i)
            line = f.readline()
    useridDict = {}
    with open('Data/user.map', 'w') as f:
        cnt = 0
        for i in useridList:
            f.writelines('\t'.join((i, str(cnt))) + '\n')
            useridDict[i] = cnt
            cnt = cnt + 1
    itemidDict = {}
    with open('Data/item.map', 'w') as f:
        cnt = 0
        for i in itemidList:
            f.writelines('\t'.join((i, str(cnt))) + '\n')
            itemidDict[i] = cnt
            cnt = cnt + 1
    print("mapping finished!")


def generateDataset(filename, num_test, num_nega):
    useridDict = {}
    useridList =[]
    cnt = 0
    with open('Data/user.map', 'r') as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            useridDict[arr[0]] = int(arr[1])
            useridList.append(int(arr[1]))
            print('user line %i: %s, %i' % (cnt, arr[0], int(arr[1])))
            cnt = cnt + 1
            line = f.readline()
    itemidDict = {}
    itemidList = []
    cnt = 0
    with open('Data/item.map', 'r') as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            itemidDict[arr[0]] = int(arr[1])
            itemidList.append(int(arr[1]))
            print('item line %i: %s, %i' % (cnt, arr[0], int(arr[1])))
            cnt = cnt + 1
            line = f.readline()
    print('get mapping finished!')
    interacitonDict = OrderedDict()
    dic = OrderedDict()
    cnt = 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i, l = useridDict[arr[0]], itemidDict[arr[1]], int(arr[2])
            print('line %i: %i, %i, %i' % (cnt, u, i, l))
            cnt = cnt + 1
            if (u, i) in dic.keys():
                dic[(u, i)] += l
            else:
                dic[(u, i)] = l

            if u in interacitonDict.keys():
                interacitonDict[u].append(i)
            else:
                interacitonDict[u] = []
                interacitonDict[u].append(i)
            line = f.readline()
    num_users = len(useridList)
    print("step1 finished!")
    negativeDict = OrderedDict()
    testList = []
    for u in useridList:
        le = len(interacitonDict[u])
        for cnt in range(num_test):
            i = interacitonDict[u][np.random.randint(le)]
            testList.append((u, i))
            dic.pop((u, i))
            interactionList = interacitonDict[u]
            negativeDict[(u, i)] = []
            negativeList = list(set(itemidList) - set(interactionList))
            for cnt in range(num_nega):
                negativeDict[(u, i)].append(negativeList[np.random.randint(len(negativeList))])

    print("step2 finished!")
    with open('Data/tafeng.train.rating', 'w') as f:
        for (u, i) in dic.keys():
            f.writelines('\t'.join((str(u), str(i), str(dic[(u, i)]))) + '\n')

    with open('Data/tafeng.test.rating', 'w') as f:
        for (u, i) in testList:
            f.writelines('\t'.join((str(u), str(i))) + '\n')

    with open('Data/tafeng.test.negative', 'w') as f:
        for (u, i) in negativeDict.keys():
            line = []
            line.append('('+str(u)+','+str(i)+')')
            for id in negativeDict[(u, i)]:
                line.append(str(id))
            f.writelines('\t'.join(line) + '\n')

    print("Dataset generated!")



if __name__ == '__main__':
    # mapUserandItem('Data/tafeng_all_data.rating')
    # preprocess('Data/ta_feng_all_months_merged.csv')
    generateDataset('Data/tafeng_all_data.rating', 1, 100)
    pass