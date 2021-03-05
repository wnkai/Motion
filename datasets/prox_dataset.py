import os
import torch
import pickle
from torch.utils.data import Dataset
from rich.progress import track

Missing_files = ['MPH1Library_00145_01', 'MPH1Library_03301_01',]
class ProxData(Dataset):
    def __init__(self, args):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        PROXD_DIR = args.proxd_dir
        print(device, PROXD_DIR)

        all_param = []
        for dir_name in track(sequence = sorted(os.listdir(PROXD_DIR)),
                              description ='Loading PROXD Dataset...',):
            # ignore missing prox files
            if dir_name in Missing_files:
                continue

            tmp = {'name': dir_name, 'pkl_datas':[]}
            fullpath = os.path.join(PROXD_DIR, dir_name)
            fullpath = fullpath + '/results'
            for file_name in os.listdir(fullpath):
                pkl_path = fullpath +'/'+ file_name + '/000.pkl'
                with open(pkl_path, 'rb') as f:
                    param = pickle.load(f, encoding='latin1')
                tmp['pkl_datas'].append(param)

            all_param.append(tmp)


        self.windows = self.make_windows(args, all_param)

        self.length = len(self.windows)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        param = self.windows[item]
        name, datas = param['name'], param['datas']

        seq = []
        for pkl in datas:
            body_pose = torch.tensor(pkl['body_pose'])
            betas = torch.tensor(pkl['betas'])
            tmp = torch.cat([body_pose, betas], 1)
            seq.append(tmp)
        seq = torch.stack(seq, 1).squeeze()

        num_nan = torch.sum(torch.isnan(seq))

        if num_nan != 0:
            #print(num_nan)
            seq = torch.where(torch.isnan(seq), torch.full_like(seq, 0), seq)

        return seq

    @staticmethod
    def make_windows(args, all_param):
        WINDOWS_SIZE = args.windows_size
        windows = []

        for scence in track(sequence = all_param,
                            description ='Making Windows...', ):
            name = scence['name']
            length = len(scence['pkl_datas'])
            datas = scence['pkl_datas']

            for i in range(length // WINDOWS_SIZE):
                first_idx = i * WINDOWS_SIZE
                tmp = {'name': name, "frame":[first_idx, first_idx+WINDOWS_SIZE-1], 'datas': []}
                for j in range(WINDOWS_SIZE):
                    index = first_idx + j
                    tmp['datas'].append(datas[index])
                windows.append(tmp)

        return windows