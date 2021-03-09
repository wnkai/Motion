from datasets.prox_dataset import ProxData
from datasets.amass_dataset import AMASSData

def create_PROXDdataset(args):
    return ProxData(args, slice = True)

def create_PROXDdataset_noslice(args):
    return ProxData(args, slice = False)

def create_AMASSdataset(args):
    return AMASSData(args)