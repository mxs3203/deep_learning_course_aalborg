

import torch


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size).cuda()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.fc5 = torch.nn.Linear(self.hidden_size, self.output_size).cuda()
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out

