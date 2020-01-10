import torch
import torch.nn as nn
from time import time
from evaluate import *
from Dataset import DatasetLoder

class GMFModel(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMFModel, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_users, latent_dim)
        self.MF_Embedding_Item = nn.Embedding(num_items, latent_dim)
        self.prediction_layer = nn.Linear(latent_dim, 1)
        self.logistic = nn.Sigmoid()
        self._init_weigth_()

    def forward(self, userinput, iteminput):
        user_latent = self.MF_Embedding_User(userinput)
        item_latent = self.MF_Embedding_Item(iteminput)
        #user_latent = userinput.view(userinput.size()[0], -1)  # 相当于keras的flatten（）
        #item_latent = iteminput.view(iteminput.size()[0], -1)
        predict_vector = user_latent * item_latent
        predict = self.prediction_layer(predict_vector)
        prediction = self.logistic(predict)
        return prediction.view(-1)

    def _init_weigth_(self):
        nn.init.normal_(self.MF_Embedding_User.weight, std=0.01)
        nn.init.normal_(self.MF_Embedding_Item.weight, std=0.01)
        # 疑问：nonlinearity参数的意思
        # 将均匀分布生成值输入到nonlinearity中得到的值再放到权重矩阵中？
        # 还是均匀分布生成值输入到权重矩阵中，该层的激活函数是nonlinearity？
        nn.init.kaiming_uniform_(self.prediction_layer.weight, a=1, nonlinearity='sigmoid')


def use_optimizer(model, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                                          lr=params['lr'],
                                                          weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer

mf_config = {
    'epochs': 20,
    'latent_dim': 200,
    'K': 10,
    'path': 'Data/tafeng',
    'dataset': 'tafeng',
    'batchsize': 256,
    'regs': [0, 0],
    'num_neg': 4,
    'optimizer': 'adam',
    'lr': 0.001,
    'l2_reg': 0,
    'verbose': 1,
    'out': 1,
    'modelpath': 'Pretrain/'

}

if __name__ == '__main__':
    print('start!')
    datasetloader = DatasetLoder(mf_config['path'])
    print('usernum: %i' % (datasetloader.num_users))
    print('itemnum: %i' % (datasetloader.num_items))
    trainloader = datasetloader.get_train_loader(mf_config['batchsize'], mf_config['num_neg'])
    testRatings = datasetloader.testRatings
    testNegatives = datasetloader.testNegatives
    print('load Data finished!')

    latent_dim = mf_config['latent_dim']
    epochs = mf_config['epochs']
    K = mf_config['K']
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' % (mf_config['dataset'], mf_config['latent_dim'], time())

    GMF = GMFModel(datasetloader.num_users, datasetloader.num_items, latent_dim)
    loss_function = nn.BCELoss()
    optimizer = use_optimizer(GMF, mf_config)

    (hits, ndcgs) = evaluate_model(GMF, testRatings, testNegatives, K)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    for epoch in range(epochs):
        t1 = time()

        GMF.train()
        total_loss = 0
        for user, item, label in trainloader:
            GMF.zero_grad()
            prediction = GMF(user, item)
            loss = loss_function(prediction, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        GMF.eval()

        t2 = time()

        (hits, ndcgs) = evaluate_model(GMF, testNegatives, testNegatives, K)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if mf_config['out'] > 0:
                torch.save(GMF, model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
