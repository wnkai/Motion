import os
import torch
import torch.optim
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.model_save_dir = os.path.join(args.save_dir, 'models')

    @abstractmethod
    def set_input(self, inputdata):
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass