from torch.utils import data
import torch
from datasets.amass_dataset import AMASSData
from datasets.mixamo_dataset import MIXAMOData


class MixedData0(data.Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()

        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0 or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]

class MixedData(data.Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, args):
        self.mixamo = MIXAMOData(args, 'JEAN')
        self.amass = AMASSData(args)

        self.length = max(self.amass.__len__(), self.mixamo.__len__())

    def __len__(self):

        return self.length

    def __getitem__(self, item):
        amass_data = self.amass.__getitem__(item)
        mixamo_data = self.mixamo.__getitem__(item)

        return [amass_data, mixamo_data]