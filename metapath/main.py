#统一pytorch版本
#得分rmse, 推荐：recall,precision,NDCG,HR
import utils
import torch
from torch import optim
import numpy as np

batch_size = 512
epochs = 300
embed_size = 64
learning_rate = 0.1 #0.01-0.001
lambda_0 = 0.05 #日志保存，training参数，训练速度调优，log的时间
lambda_1 = 0.05

num_user, num_movie, movie_dataset, Y, R, MD, user_index, movie_index, label= utils.load_data()
print('num_user:',num_user)
print('num_movie:',num_movie)
S = utils.pathsim(MD)
L = utils.laplacian(S)
print('L has done')

Y = torch.from_numpy(Y)
R = torch.from_numpy(R)
L = torch.from_numpy(L)

U = torch.randn((num_user, embed_size), requires_grad=True)
V = torch.randn((num_movie, embed_size), requires_grad=True)

torch.nn.init.kaiming_normal_(U)
torch.nn.init.kaiming_normal_(V)

optimizer = optim.SGD([U,V], lr=learning_rate)

#loss_1 = torch.norm(Y*(R-torch.mm(U, V.t())))+lambda_0*(torch.norm(U)+torch.norm(V))
#index_array = np.arange(embed_size)
#index = torch.from_numpy(index_array)
#解决torch下迹的问题
#loss_2 = lambda_1* torch.sum(torch.diag(torch.mm(torch.mm(V.t(), L),V))).float()#.gatter(0,index))
#loss = loss_1 +loss_2
#loss = loss_1
for epoch in range(epochs):
    loss_1 = torch.norm(Y * (R - torch.mm(U, V.t()))) + lambda_0 * (torch.norm(U) + torch.norm(V))
    loss_2 = lambda_1 * torch.trace(torch.mm(torch.mm(V.t().float(), L.float()), V.float()))
    loss = loss_1 +loss_2
    #loss = loss_1
    optimizer.zero_grad()
    loss.backward()
    print('train epoch:', epoch, 'loss:', loss)
    if (epoch+1)%10 ==0:
        rmse = utils.evaluation(U, V, user_index, movie_index, label)
        print('rmse:', rmse)
    optimizer.step()









