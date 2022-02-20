import torch
from torch import nn, optim

IMAGE_SIZE = 784
FIRST_LAYER = 128
SECOND_LAYER = 64
THIRD_LAYER = 10
FOURTH_LAYER = 10
FIFTH_LAYER = 10
LABELS = 10
LEARNING_RATE = 0.005
import torch.nn.functional as F


class ModelE(nn.Module):
    def __init__(self):
        super(ModelE, self).__init__()
        self.__fc0 = nn.Linear(IMAGE_SIZE, FIRST_LAYER)
        self.__fc1 = nn.Linear(FIRST_LAYER, SECOND_LAYER)
        self.__fc2 = nn.Linear(SECOND_LAYER, THIRD_LAYER)
        self.__fc3 = nn.Linear(THIRD_LAYER, FOURTH_LAYER)
        self.__fc4 = nn.Linear(FOURTH_LAYER, FIFTH_LAYER)
        self.__fc5 = nn.Linear(FIFTH_LAYER, LABELS)
        self.__optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)



    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.__fc0(x))
        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        x = F.relu(self.__fc3(x))
        x = F.relu(self.__fc4(x))
        return F.log_softmax(self.__fc5(x), dim=-1)

    def get_optimizer(self):
        return self.__optimizer

    def set_optimizer(self, opti):
        self.__optimizer = opti
