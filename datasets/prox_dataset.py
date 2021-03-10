import os
import torch
import pickle
import torch.utils.data as data
from rich.progress import track

Missing_files = ['MPH1Library_00145_01', 'MPH1Library_03301_01',]
class ProxData(data.Dataset):
    def __init__(self, args, slice = True):
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
            for file_name in sorted(os.listdir(fullpath)):
                pkl_path = fullpath +'/'+ file_name + '/000.pkl'
                with open(pkl_path, 'rb') as f:
                    param = pickle.load(f, encoding='latin1')
                tmp['pkl_datas'].append(param)

            all_param.append(tmp)


        if slice:
            self.windows = self.make_windows(args, all_param)
        else:
            self.windows = self.prepare(args, all_param)

        self.length = len(self.windows)


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        param = self.windows[item]
        name, datas = param['name'], param['datas']

        seq_pose = []
        seq_static = []

        for pkl in datas:
            body_pose = torch.tensor(pkl['body_pose'])
            tmp_pose = torch.cat([body_pose], -1)
            seq_pose.append(tmp_pose)

            betas = torch.tensor(pkl['betas'])
            root_orient = torch.tensor(pkl['global_orient'])
            root_trans = torch.tensor(pkl['transl'])
            tmp_static = torch.cat([betas, root_orient, root_trans], -1)
            seq_static.append(tmp_static)

        def deart(seq):
            seq = torch.stack(seq, 0).squeeze()
            num_nan = torch.sum(torch.isnan(seq))
            if num_nan != 0:
                seq = torch.where(torch.isnan(seq), torch.full_like(seq, 0), seq)
            seq = seq.permute(1, 0)
            seq = seq.float()
            return seq

        seq_pose = deart(seq_pose)
        seq_static = deart(seq_static)

        return seq_pose, seq_static


    def get_noslice(self, item):
        scence_name = self.windows[item]['name']
        scence_name = scence_name[:scence_name.find('_')]

        seq_pose, seq_static = self.__getitem__(item)

        seq_pose = seq_pose.reshape([1, *seq_pose.shape])
        seq_static = seq_static.reshape([1, *seq_static.shape])

        return [seq_pose, seq_static], scence_name

    @staticmethod
    def prepare(args, all_param):
        windows = []
        for scence in track(sequence=all_param,
                            description='Making Windows...', ):
            name = scence['name']
            length = len(scence['pkl_datas'])
            datas = scence['pkl_datas']
            tmp = {'name': name, "frame": [0, length], 'datas': []}

            for i in range(length):
                tmp['datas'].append(datas[i])

            windows.append(tmp)

        return windows

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