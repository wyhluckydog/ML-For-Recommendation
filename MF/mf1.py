import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.cuda.set_device(0)
df=pd.DataFrame(pd.read_csv('ratings.csv',header=1))
df=df.sample(frac=1).reset_index(drop=True)
#print(df)
cut_idx = int(round(0.2 * df.shape[0]))
df_test, df_train = df.loc[:cut_idx], df.loc[cut_idx:]
#print(df_train)
df_train.index=df_train.index-cut_idx
user_size = len(df['1'].unique())
item_size = len(df['3'].unique())
user_dict = {x: i for i, x in enumerate(df['1'].unique())}
#建立映射，就想象一个评分矩阵，然后这个user_dict里面是，userid到评分矩阵第几行的映射
print(len(user_dict))

item_dict={x: i for i, x in enumerate(df['3'].unique())}
#道理同user_dict
print(item_dict[6])

##data_matrix=np.zeros((user_size+1,item_size+1),dtype=float)




class mf(nn.Module):
    def __init__(self,user_size,item_size):
        super().__init__()
        dim=100#维度20
        self.W = nn.Parameter(torch.rand(user_size+1, dim)) # 用户隐藏层
        self.H = nn.Parameter(torch.rand(item_size+1, dim)) # 商品隐藏层
        #self.Bu = nn.Parameter(torch.rand(user_size + 1, 1))  # 用户偏置层
        #self.Bi = nn.Parameter(torch.rand(item_size + 1, 1))  # 商品偏置层
        self.U=nn.Parameter(torch.rand(1,1))

    def forward(self, u, i, r):
        error = 0
        #print(u)
        #print(i)
        user_id=[]
        item_id=[]
        ratings=[]
        for ylz in range(1, len(u)):
            user_id.append(user_dict[u[ylz]])
            item_id.append(item_dict[i[ylz]])
            ratings.append(r[ylz])
        ##获取batch里面的user索引、item索引、评分
        predict_val = torch.mul(self.W[user_id, :], self.H[item_id, :]).sum(1)
        ##选取这些行的每一个乘对应列的然后求和，得到预测值
        predict_val=torch.unsqueeze(predict_val, 1)
        ##转置（其实这里有点憨憨，因为那个偏置是竖着的。然后得到的predict_val是横着的）
        #predict_val = predict_val + self.Bu[user_id] + self.Bi[item_id]+self.U
        ratings=torch.tensor(ratings,device='cuda:0')
        ratings=torch.unsqueeze(ratings, 1)
        error=torch.sub(predict_val,ratings)

        error=torch.mul(error,error).sum()
        ##这个二范数有点，emmmm感觉不太对，我这个是矩阵的二范数，不是对应的向量的二范数
        ##error=error+0.1*torch.norm(self.Bu[user_id],p=2)+0.1*torch.norm(self.Bi[item_id],p=2)
        #error = error + (torch.sqrt((abs(predict_val - ratings))))

        error = error / len(u)
        #print("error")
        #print(error)
        #print("矩阵")
        #print(self.W)
        ##print(self.H)
        return error




model = mf(user_size, item_size).cuda()
optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
enpoch=0
picture_train=[]
picture_test=[]
batch=1280
#smooth_loss=[]
#smooth_loss_test=[]
while enpoch <200:
    smooth_loss = []
    smooth_loss_test = []
    enpoch=enpoch+1
    ##训练集
    for i in range(0,len(df_train['1'])//batch):
        u=[]
        t=[]
        r=[]
        u=(df_train.loc[i * batch:i * batch +batch,'1'])
        u.index=u.index-batch*i
        ##print(u)
        t=(df_train.loc[i * batch:i * batch + batch, '3'])
        t.index=t.index-batch*i
        r = (df_train.loc[i * batch:i * batch + batch, '4'])
        r.index = r.index - batch * i
        ##u表示选取batch里面的user，t就是item的id，r就是选取的这些列的rating
        optimizer.zero_grad()
        loss = model(u, t, r).cuda()
        loss.backward()
        optimizer.step()

        smooth_loss.append(loss.item())
    #smooth_loss=smooth_loss/(1+len(df_train['1'])//batch)
##测试集
    for i in range(0,len(df_test['1'])//batch):
        ##batch
        u=[]
        t=[]
        u=(df_test.loc[i * batch:i * batch +batch,'1'])
        u.index=u.index-batch*i
        ##print(u)
        t=(df_train.loc[i * batch:i * batch + batch, '3'])
        t.index=t.index-batch*i
        r = (df_train.loc[i * batch:i * batch + batch, '4'])
        r.index = r.index - batch * i

        optimizer.zero_grad()
        loss_test = model(u, t, r).cuda()
        smooth_loss_test.append(loss_test.item())
    #smooth_loss_test=smooth_loss_test/(1+len(df_test['1'])//batch)


        ##smooth_loss=0.99*smooth_loss+0.01*loss
        #print(smooth_loss.item())
        #smooth_loss=np.array(smooth_loss.item())
    print(enpoch)
    print("loss")

    #print(smooth_loss_test)
    if enpoch>1:
        picture_train.append(np.mean(smooth_loss))
        picture_test.append(np.mean(smooth_loss_test))
    print(np.mean(smooth_loss))
    print("test")
    print(np.mean(smooth_loss_test))

    #print(smooth_loss)
plt.clf()
plt.plot(range(len(picture_train)), picture_train,label='Training Data')
plt.plot(range(len(picture_test)), picture_test,label='Test Data')
plt.title('The MovieLens Dataset Learning Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()
    

