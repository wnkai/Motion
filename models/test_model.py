import torch
from torch import nn
import torch.nn.functional as F

class test_model(nn.Module):
    def __init__(self, args):
        super(test_model, self).__init__()
        self.WINDOWS_SIZE = args.windows_size
        self.fc1 = nn.Linear(73*self.WINDOWS_SIZE, 2048)
        self.fc4 = nn.Linear(2048, 73*self.WINDOWS_SIZE)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = torch.reshape(x,(-1,self.WINDOWS_SIZE,73,))

        return x