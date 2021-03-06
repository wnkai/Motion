import os
import torch
import numpy as np
import torch.utils.data as data
from rich.progress import track
import torch.nn.functional as F
import datasets.smplx_topology

class AMASSData(data.Dataset):
    def __init__(self, args):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        AMASS_DIR = args.amass_dir
        print(device, AMASS_DIR)
        self.enableDataSet = ['BioMotionLab_NTroje', 'BMLmovi', 'BMLhandball', 'MPI_HDM05', 'CMU', 'SFU']
        #self.enableDataSet = ['SFU', 'CMU']

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
                    if len(param_file) < 6:
                        continue

                    def down_sampling(inputs, rate):
                        channel = inputs.shape[1]
                        trans = torch.Tensor(inputs).reshape(1, -1, channel).permute(0, 2, 1)
                        trans_int = F.interpolate(trans, scale_factor = args.standard_framerate / rate).permute(0, 2, 1)
                        trans_int = trans_int.reshape(-1, channel)
                        return trans_int.numpy()


                    frame_rate = float(param_file['mocap_framerate'])
                    param = {
                        'dmpls': down_sampling(param_file['dmpls'], frame_rate),
                        'trans': down_sampling(param_file['trans'], frame_rate),
                        'poses': down_sampling(param_file['poses'], frame_rate),

                        'mocap_framerate': param_file['mocap_framerate'],
                        'betas': param_file['betas'],
                        'gender': param_file['gender'],
                    }

                    param['length'] = param['trans'].shape[0]
                    tmp['npy_datas'] = param #sequence
                    all_param.append(tmp)

        self.windows = self.make_windows(args, all_param)
        self.length = len(self.windows)

        self.offset = torch.zeros([1,datasets.smplx_topology.JOINT_NUM* 3, 1])
        #self.mean_pose, self.var_pose, self.mean_static, self.var_static = self.get_mean_var()

    def get_mean_var(self):
        poses, statics = [], []
        for i in range(self.length):
            pose, static = self.__getitem__(i)
            poses.append(pose.permute(1,0))
            statics.append(static.permute(1,0))

        poses, statics = torch.stack(poses, dim=0).reshape(-1, 63), torch.stack(statics, dim=0).reshape(-1, 16)
        mean_pose = torch.mean(poses,dim=0)
        var_pose = torch.var(poses,dim=0)

        mean_static = torch.mean(statics,dim=0)
        var_static = torch.var(statics,dim=0)

        return mean_pose, var_pose, mean_static, var_static

    def normalize(self, inputs):
        seq_pose, seq_static = inputs

        seq_pose = seq_pose.permute(0, 2, 1)
        shape_pose = seq_pose.shape
        seq_pose = seq_pose.reshape(-1, 63)
        org_var_pose = torch.sqrt(self.var_pose)
        seq_pose  = (seq_pose - self.mean_pose) / org_var_pose
        seq_pose = seq_pose.reshape(shape_pose)
        seq_pose = seq_pose.permute(0, 2, 1)

        seq_static = seq_static.permute(0, 2, 1)
        shape_static = seq_static.shape
        seq_static = seq_static.reshape(-1, 16)
        org_var_static = torch.sqrt(self.var_static)
        seq_static = (seq_static - self.mean_static) / org_var_static
        seq_static = seq_static.reshape(shape_static)
        seq_static = seq_static.permute(0, 2, 1)

        return seq_pose, seq_static

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        from scipy.spatial.transform import Rotation as R
        param = self.windows[item]
        name, datas = param['name'], param['datas']

        seq_pose = []
        seq_static = []

        for pkl in datas:
            body_pose = torch.tensor(pkl['poses']).reshape(-1, 3)
            body_pose_zeros = torch.zeros_like(body_pose)
            body_pose_zeros[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :] = body_pose[
                                                                                                             [0, 3, 6,
                                                                                                              9, 1, 4,
                                                                                                              7, 10, 2,
                                                                                                              5, 8, 11,
                                                                                                              14, 12,
                                                                                                              15, 17,
                                                                                                              19, 13,
                                                                                                              16, 18,
                                                                                                              20], :]
            r = R.from_rotvec(body_pose)
            euler = r.as_quat()
            euler = euler[:,[3,0,1,2]]
            body_pose = torch.Tensor(euler).reshape(-1)

            root_trans = torch.tensor(pkl['root_trans'])
            tmp_pose = torch.cat([body_pose, root_trans], -1)
            seq_pose.append(tmp_pose)

            betas = torch.tensor(pkl['betas'])
            root_orient = torch.tensor(pkl['root_orient'])
            root_trans = torch.tensor(pkl['root_trans'])
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
                        'root_orient': data['poses'][index][0:3],
                        'poses': data['poses'][index][3:66],
                    })
                windows.append(tmp)

        return windows
