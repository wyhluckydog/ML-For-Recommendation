import torch.optim as optim

class Optimizer:
    def __init__(self, params, lr=.05, weight_decay=0, optimizer_type='Adagrad'):
        self.optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()