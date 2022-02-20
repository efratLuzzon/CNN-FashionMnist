import torch
from torch import nn, optim


IMAGE_SIZE = 784
FIRST_LAYER = 100
SECOND_LAYER = 50
LABELS = 10
LEARNING_RATE = 0.01
import torch.nn.functional as F


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.__fc0 = nn.Linear(IMAGE_SIZE, FIRST_LAYER)
        self.__fc1 = nn.Linear(FIRST_LAYER, SECOND_LAYER)
        self.__fc2 = nn.Linear(SECOND_LAYER, LABELS)
        self.bn1 = nn.BatchNorm1d(FIRST_LAYER)
        self.bn2 = nn.BatchNorm1d(SECOND_LAYER)
        self.__optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.bn1(self.__fc0(x)))
        x = F.relu(self.bn2(self.__fc1(x)))
        return F.log_softmax(self.__fc2(x), dim=-1)

    def get_optimizer(self):
        return self.__optimizer

    def set_optimizer(self, opti):
        self.__optimizer = opti