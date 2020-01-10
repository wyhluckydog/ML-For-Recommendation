import torch
import numpy as np
import heapq
import math

def evaluate_model(model, testRatings, testNegatives, K):

    hits, ndcgs = [], []
    for idx in range(len(testRatings)):
        hr, ndcg = eval_one_rating(idx, model, testRatings, testNegatives, K)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def eval_one_rating(idx, model, testRatings, testNegatives, K):
    rating = testRatings[idx]
    items = testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)

    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    with torch.no_grad():
        testusers = torch.LongTensor(users)
        testitems = torch.LongTensor(items)
        predictions = model(testusers, testitems)

    predictions = predictions.data.view(-1).tolist()
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
