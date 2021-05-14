

import torch
import torch.nn as nn


class MyRNN(torch.nn.Module):

    def __init__(self,hidden_dim):
        super(MyRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(101, hidden_dim, 1, batch_first=True)
        self.rnn2 = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 3)


    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_().cuda()
        x, hidden = self.rnn(x, h0)
        x, hidden = self.rnn2(x, hidden)
        x = x[:, -1, :]
        print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x, hidden