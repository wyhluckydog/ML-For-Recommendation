import torch
import numpy as np

def get_recall(indices, target):
    targets = target.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall

def get_F1(indices, target):
    f1 = get_recall(indices, target)    #单个预测情况，precision和recall相等
    return f1


def get_mrr(indices, targets):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1 #每个结果命中的位置
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)    #计算每个元素的倒数 1/x
    mrr = torch.sum(rranks).data / targets.size(0)  #计算平均
    return mrr

#感觉不合适,未使用
def get_NDCG(indices, targets): #传入每个session的下一个item的预测，和真实标签
    batch_size = len(targets)
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1]
    ranks = ranks.cpu()
    NDCG = (np.log2(2) / np.log2(ranks + 2)).sum() / batch_size #计算NDCG
    return NDCG

def evaluate(indices, targets, k=20):
    _, indices = torch.topk(indices, k, -1) #取每行前20个值
    f_one = get_F1(indices, targets)
    mrr = get_mrr(indices, targets)
    return f_one, mrr
