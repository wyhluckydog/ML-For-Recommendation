import os
import lib
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.evaluation = lib.Evaluation(self.model, self.loss_func, use_cuda, k=20)

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        epoch_num = []
        train_mrrs = []
        test_mrrs = []

        for epoch in range(start_epoch, end_epoch + 1):
            print('Start training in Epoch #', epoch)
            epoch_num.append(np.int(epoch))
            train_loss = self.train_epoch(epoch) #训练完一轮

            print("Evaluation testset in Epoch #", epoch)
            test_loss, test_f1, test_mrr = self.evaluation.eval(self.eval_data, self.batch_size)
            print("Evaluation trainset in Epoch #", epoch)
            train_loss_eval, train_f1, train_mrr = self.evaluation.eval(self.eval_data, self.batch_size)
            test_mrrs.append(np.float(test_mrr))
            train_mrrs.append(np.float(train_mrr))

            print("Epoch: {}, train_loss: {:.4f}, time: {}".format(epoch, train_loss, time.time()))
            checkpoint = {
                'model': self.model,
                'epoch': epoch,
                'optim': self.optim,
            }
            model_name = os.path.join("C:/Users/34340/Desktop/rsc15_dataset/", "model_{}.pt".format(epoch))   #服务器位置
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)
            if(epoch % 5 == 0):
                print("Epoch: {}, test_loss = {:.4f}, f1 = {:.2f}, mrr = {:.2f}".format(epoch, test_loss, test_f1, test_mrr))
                plt.clf()   #clean the plot
                plt.plot(epoch_num, test_mrrs)
                plt.plot(epoch_num, train_mrrs)
                plt.xlabel("Number of Epochs")
                plt.ylabel("MRR")
                plt.legend(['Test Data', 'Training Data'])
                if(epoch == end_epoch):
                    plt.show(block=True)
                else:
                    plt.show(block=False)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        dataloader = lib.DataLoader(self.train_data, self.batch_size)
        #iterate reading batch_size items from the dataset and put into the GRU
        for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters=100):
            input = input.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = self.model(input, hidden) #logit(B,item)  hidden(num_layers,B,H)
            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses