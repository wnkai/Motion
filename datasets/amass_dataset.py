import os
import torch
import numpy as np
from torch.utils.data import Dataset
from rich.progress import track

class AMASSData(Dataset):
    def __init__(self, args):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        AMASS_DIR = args.amass_dir
        print(device, AMASS_DIR)
        self.enableDataSet = ['CMU',]
        all_param = []

        for dir_name in track(sequence = sorted(self.enableDataSet),
                              description ='Loading AMASS Dataset...',):
            dataset_path = os.path.join(AMASS_DIR, dir_name)
            for subject_name in sorted(os.listdir(dataset_path)):
                subject_path = os.path.join(dataset_path, subject_name)
                for file_name in sorted(os.listdir(subject_path)):
                    file_path = os.path.join(subject_path, file_name)

                    tmp = {'name': file_path, 'npy_datas': None}

                    param_file = np.load(file_path, encoding='latin1')
                    #print(file_path,len(param_file))
                    if len(param_file) != 6:
                        continue
                    param = {
                        'length': param_file['trans'].shape[0],
                        'betas': param_file['betas'],
                        'gender': param_file['gender'],
                        'dmpls': param_file['dmpls'],
                        'mocap_framerate': param_file['mocap_framerate'],
                        'trans': param_file['trans'],
                        'poses': param_file['poses'],
                    }
                    tmp['npy_datas'] = param #sequence

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
            trans = torch.tensor(pkl['root_trans'])
            body_pose = torch.tensor(pkl['poses'])
            betas = torch.tensor(pkl['betas'])
            tmp = torch.cat([body_pose, trans], -1)
            seq.append(tmp)
        seq = torch.stack(seq, 0).squeeze()
        seq = seq.permute(1,0)
        seq = seq.float()
        return seq

    @staticmethod
    def make_windows(args, all_param):
        WINDOWS_SIZE = args.windows_size
        windows = []

        for motion in track(sequence=all_param,
                            description='Making Windows...', ):
            name = motion['name']
            data = motion['npy_datas']
            length = data['length']

            for i in range(length // WINDOWS_SIZE):
                first_idx = i * WINDOWS_SIZE
                tmp = {'name': name,
                       "frame":[first_idx, first_idx+WINDOWS_SIZE-1],

                       'gender': data['gender'],
                       #'dmpls': data['dmpls'],
                       #'mocap_framerate': data['mocap_framerate'],
                       'datas': []
                       }
                for j in range(WINDOWS_SIZE):
                    index = first_idx + j
                    tmp['datas'].append({
                        'betas': data['betas'][:10],
                        'root_trans': data['trans'][index][:3],
                        'poses': data['poses'][index][3:66],
                    })
                windows.append(tmp)

        return windows
