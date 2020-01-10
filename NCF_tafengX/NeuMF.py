import torch
import torch.nn as nn
from time import time
from evaluate import *
from Dataset import DatasetLoder
from MLP import MLPModel
from GMF import GMFModel
class NeuMFModel(nn.Module):
    def __init__(self, num_users, num_items, neumf_config):
        super(NeuMFModel, self).__init__()
        mf_dim = neumf_config['latent_dim']
        layers = neumf_config['layers']

        self.MF_Embedding_User = nn.Embedding(num_users, mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_items, mf_dim)
        self.MLP_Embedding_User = nn.Embedding(num_users, layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_items, layers[0] // 2)

        # MF part

        # MLP part
        num_layers = len(layers)
        MLP_modules = []
        for i in range(1, num_layers):
            input_size = layers[i - 1]
            out_size = layers[i]
            MLP_modules.append(nn.Linear(input_size, out_size))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.prediction_layer = nn.Linear(layers[-1] + mf_dim, 1)
        self.logistic = nn.Sigmoid()
        if neumf_config['pretrain'] == 1:
            self.GMF = torch.load(neumf_config['mf_pretrain'])
            self.MLP = torch.load(neumf_config['mlp_pretrain'])
            self._load_preweight_()
        else:
            self._init_weight_()

    def forward(self, userinput, iteminput):
        mf_user_latent = self.MF_Embedding_User(userinput)
        mf_item_latent = self.MF_Embedding_Item(iteminput)
        mf_vector = mf_user_latent * mf_user_latent

        user_latent = self.MLP_Embedding_User(userinput)
        item_latent = self.MLP_Embedding_Item(iteminput)
        mlp_latent = torch.cat([user_latent, item_latent], dim=-1)  # the concat latent vector
        mlp_vector = self.MLP_layers(mlp_latent)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        predict = self.prediction_layer(predict_vector)
        prediction = self.logistic(predict)
        return prediction.view(-1)

    def _init_weight_(self):
        nn.init.normal_(self.MF_Embedding_User.weight, std=0.01)
        nn.init.normal_(self.MF_Embedding_Item.weight, std=0.01)

        nn.init.normal_(self.MLP_Embedding_User.weight, std=0.01)
        nn.init.normal_(self.MLP_Embedding_Item.weight, std=0.01)

        # 疑问：nonlinearity参数的意思
        # 将均匀分布生成值输入到nonlinearity中得到的值再放到权重矩阵中？
        # 还是均匀分布生成值输入到权重矩阵中，该层的激活函数是nonlinearity？
        # nn.init.kaiming_uniform_(self.prediction_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.kaiming_uniform_(self.prediction_layer.weight, a=1, nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def _load_preweight_(self):
        torch.load
        # embedding layers
        self.MF_Embedding_User.weight.data.copy_(
            self.GMF.MF_Embedding_User.weight)
        self.MF_Embedding_Item.weight.data.copy_(
            self.GMF.MF_Embedding_Item.weight)
        self.MLP_Embedding_User.weight.data.copy_(
            self.MLP.MLP_Embedding_User.weight)
        self.MLP_Embedding_Item.weight.data.copy_(
            self.MLP.MLP_Embedding_Item.weight)

        # mlp layers
        for (m1, m2) in zip(
                self.MLP_layers, self.MLP.MLP_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)

        # predict layers
        predict_weight = torch.cat([
            self.GMF.prediction_layer.weight,
            self.MLP.prediction_layer.weight], dim=1)
        predict_bias = self.GMF.prediction_layer.bias + self.MLP.prediction_layer.bias

        self.prediction_layer.weight.data.copy_(0.5 * predict_weight)
        self.prediction_layer.bias.data.copy_(0.5 * predict_bias)



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

neumf_config = {
    'epochs': 20,
    'latent_dim': 200,
    'K': 10,
    'path': 'Data/tafeng',
    'dataset': 'tafeng',
    'batchsize': 256,
    'layers': [64, 32, 16, 8],
    'reg_layers': [0, 0, 0, 0],
    'num_neg': 4,
    'optimizer': 'adam',
    'lr': 0.001,
    'l2_reg': 0,
    'verbose': 1,
    'out': 1,
    'modelpath': 'Pretrain/',
    'pretrain': 1,
    'mf_pretrain': 'Pretrain/tafeng_GMF_200_1573707343.h5',
    'mlp_pretrain': 'Pretrain/tafeng_MLP_200_1573716488.h5'
}


if __name__ == '__main__':
    print('NeuMF start!')
    datasetloader = DatasetLoder(neumf_config['path'])
    trainloader = datasetloader.get_train_loader(neumf_config['batchsize'], neumf_config['num_neg'])
    testRatings = datasetloader.testRatings
    testNegatives = datasetloader.testNegatives
    print('load Data finished!')

    latent_dim = neumf_config['latent_dim']
    epochs = neumf_config['epochs']
    K = neumf_config['K']
    model_out_file = 'Pretrain/%s_NeuMF_%d_%d.h5' % (neumf_config['dataset'], neumf_config['latent_dim'], time())

    NeuMF = NeuMFModel(datasetloader.num_users, datasetloader.num_items, neumf_config)
    loss_function = nn.BCELoss()
    optimizer = use_optimizer(NeuMF, neumf_config)


    (hits, ndcgs) = evaluate_model(NeuMF, testRatings, testNegatives, K)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    for epoch in range(epochs):
        t1 = time()

        NeuMF.train()
        total_loss = 0
        for user, item, label in trainloader:
            NeuMF.zero_grad()
            prediction = NeuMF(user, item)
            loss = loss_function(prediction, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        NeuMF.eval()

        t2 = time()

        (hits, ndcgs) = evaluate_model(NeuMF, testNegatives, testNegatives, K)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if neumf_config['out'] > 0:
                torch.save(NeuMF, model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
