import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from models.BaseModel import BaseModel


class AE_model(BaseModel):
    def __init__(self, args):
        super(AE_model, self).__init__(args)
        self.model = test_model(args).cuda()

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = torch.nn.MSELoss()

    def set_input(self, inputdata):
        self.inputdata = inputdata.cuda()

    def forward(self):
        self.outputdata = self.model(self.inputdata)

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.loss = self.criterion(self.inputdata, self.outputdata)
        self.loss.backward()
        self.optimizer.step()
        print(self.loss)


class test_model(nn.Module):
    def __init__(self, args):
        super(test_model, self).__init__()
        self.WINDOWS_SIZE = args.windows_size


    def forward(self, x: torch.Tensor):

        return x