import numpy as np
import pandas as pd
import datetime

raw_data = "C:/Users/34340/Desktop/rsc15_dataset/yoochoose-clicks(simple).dat"
after_process = "C:/Users/34340/Desktop/rsc15_dataset/"
dayTime = 86400

train = pd.read_csv(raw_data, sep=',', header=None, usecols=[0, 1, 2], dtype={0: np.int32, 1: str, 2: np.int64})
train.columns = ['SessionID', 'Time', 'ItemID']
train['Time'] = train.Time.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())

def removeShortSessions(data):
    sessionLen = data.groupby('SessionID').size()
    data = data[np.in1d(data.SessionID, sessionLen[sessionLen > 1].index)]
    return data

train = removeShortSessions(train)
itemLen = train.groupby('ItemID').size()
train = train[np.in1d(train.ItemID, itemLen[itemLen > 4].index)]
train = removeShortSessions(train)

timeMax = train.Time.max()
sessionMaxTime = train.groupby('SessionID').Time.max()
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index
sessionTest = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index
final_train = train[np.in1d(train.SessionID, sessionTrain)]
test = train[np.in1d(train.SessionID, sessionTest)]
test = test[np.in1d(test.ItemID, final_train.ItemID)]
final_test = removeShortSessions(test)
print('Training set has: ', len(final_train), 'Events, ', final_train.SessionID.nunique(), 'Sessions, and', final_train.ItemID.nunique(), 'Items\n\n')
final_train.to_csv(after_process + 'train_data.txt', sep=',', index=False)
print('Test set has: ', len(final_test), 'Events, ', final_test.SessionID.nunique(), 'Sessions, and', final_test.ItemID.nunique(), 'Items\n\n')
final_test.to_csv(after_process + 'test_data.txt', sep=',', index=False)
