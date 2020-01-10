import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1_max'):
        super(LossFunction, self).__init__()
        self._loss_fn = TOP1_max()

    def forward(self, logit):
        return self._loss_fn(logit)



class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1) #按行softmax，行和为1
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit) #取对角线，然后扩展成方阵，再取差的负数
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))    #计算loss
        return loss