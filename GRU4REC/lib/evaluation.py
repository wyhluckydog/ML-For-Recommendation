import lib
import numpy as np
import torch
from tqdm import tqdm

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        f1s = []
        mrrs = []
        dataloader = lib.DataLoader(eval_data, batch_size)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters=1):
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                f1, mrr = lib.evaluate(logit, target, k=self.topk)
                losses.append(loss.item())
                f1s.append(np.float(f1))
                mrrs.append(np.float(mrr))
        mean_losses = np.mean(losses)
        mean_f1 = np.mean(f1s)
        mean_mrr = np.mean(mrrs)

        return mean_losses, mean_f1, mean_mrr