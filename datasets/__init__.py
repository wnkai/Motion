from datasets.prox_dataset import ProxData
from datasets.amass_dataset import AMASSData
from datasets.mixamo_dataset import MIXAMOData
from datasets.MixedData import MixedData

def create_PROXDdataset(args):
    return ProxData(args, slice = True)

def create_PROXDdataset_noslice(args):
    return ProxData(args, slice = False)

def create_AMASSdataset(args):
    return AMASSData(args)

def create_MIXAMO_model(args):
    return MIXAMOData(args, 'JEAN')

def create_MixedData(args):
    return MixedData(args)