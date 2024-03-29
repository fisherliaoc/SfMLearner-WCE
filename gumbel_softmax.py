import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

def sample_gumbel(shape, eps=1e-3):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = 5*torch.tanh(logits) + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature = 1):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind.unsqueeze(1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

