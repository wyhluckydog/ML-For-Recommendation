import torch
import torch.nn as nn
from time import time
from evaluate import *
from Dataset import DatasetLoder

class MLPModel(nn.Module):
    def __init__(self, num_users, num_items, layers, ):
        super(MLPModel, self).__init__()
        self.MLP_Embedding_User = nn.Embedding(num_users, layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_items, layers[0] // 2)

        num_layers = len(layers)
        MLP_modules = []
        for i in range(1, num_layers):
            input_size = layers[i-1]
            out_size = layers[i]
            MLP_modules.append(nn.Linear(input_size, out_size))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.prediction_layer = nn.Linear(layers[num_layers-1], 1)
        self.logistic = nn.Sigmoid()
        self._init_weight_()

    def forward(self, userinput, iteminput):
        user_latent = self.MLP_Embedding_User(userinput)
        item_latent = self.MLP_Embedding_Item(iteminput)
        mlp_vector = torch.cat([user_latent, item_latent], dim=-1)  # the concat latent vector
        predict_vector = self.MLP_layers(mlp_vector)
        predict = self.prediction_layer(predict_vector)
        prediction = self.logistic(predict)
        return prediction.view(-1)

    def _init_weight_(self):
        nn.init.normal_(self.MLP_Embedding_User.weight, std=0.01)
        nn.init.normal_(self.MLP_Embedding_Item.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.kaiming_uniform_(self.prediction_layer.weight, a=1, nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


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


mlp_config = {
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
    'modelpath': 'Pretrain/'
}

if __name__ == '__main__':
    print('MLP start!')
    datasetloader = DatasetLoder(mlp_config['path'])
    trainloader = datasetloader.get_train_loader(mlp_config['batchsize'], mlp_config['num_neg'])
    testRatings = datasetloader.testRatings
    testNegatives = datasetloader.testNegatives
    print('load Data finished!')

    latent_dim = mlp_config['latent_dim']
    epochs = mlp_config['epochs']
    K = mlp_config['K']
    model_out_file = 'Pretrain/%s_MLP_%d_%d.h5' % (mlp_config['dataset'], mlp_config['latent_dim'], time())

    MLP = MLPModel(datasetloader.num_users, datasetloader.num_items, mlp_config['layers'])
    loss_function = nn.BCELoss()
    optimizer = use_optimizer(MLP, mlp_config)

    (hits, ndcgs) = evaluate_model(MLP, testRatings, testNegatives, K)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    for epoch in range(epochs):
        t1 = time()

        MLP.train()
        total_loss = 0
        for user, item, label in trainloader:
            MLP.zero_grad()
            prediction = MLP(user, item)
            loss = loss_function(prediction, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        MLP.eval()

        t2 = time()

        (hits, ndcgs) = evaluate_model(MLP, testNegatives, testNegatives, K)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if mlp_config['out'] > 0:
                torch.save(MLP, model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
