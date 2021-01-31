from datasets.prox_dataset import ProxData
from datasets.amass_dataset import AMASSData

def create_PROXDdataset(args):
    return ProxData(args)

def create_AMASSdataset(args):
    return AMASSData(args)