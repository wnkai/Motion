import numpy as np
import trimesh
import pyrender
import json
import smplx
import pickle
import torch
import argparse
import os
import glm
import matplotlib.pyplot as plt

class BodyMaker:
    def __init__(self, args):
        self.smplx = smplx.create(args.model_folder, model_type='smplx')
        self.body = trimesh.Trimesh()

    def get_mesh_bypath(self, path):
        with open(path, 'rb') as f:
            param = pickle.load(f, encoding='latin1')

        torch_param = {}
        for key in param.keys():
            torch_param[key] = torch.tensor(param[key])

        #torch_param['transl'] = torch.zeros(torch_param['transl'].shape)
        #torch_param['global_orient'] = torch.zeros(torch_param['global_orient'].shape)
        #torch_param['betas'] = torch.zeros(torch_param['betas'].shape)
        #torch_param['body_pose'] = torch.zeros(torch_param['body_pose'].shape)

        org = torch_param['global_orient'].reshape(1,3)
        output = self.smplx(body_pose = torch_param['body_pose'][:, 0:63],
                          global_orient = org,
                          betas = torch_param['betas'].reshape(1,10),
                          transl = torch_param['transl'].reshape(1,3))
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, self.smplx.faces,
                                   vertex_colors=vertex_colors)
        return tri_mesh


def main(args):


    env_mesh = trimesh.load('../datasets/PROX/scenes/BasementSittingBooth.ply')
    bm = BodyMaker(args=args)

    with open("/home/kaiwang/Documents/DataSet/PROX/cam2world/BasementSittingBooth.json", 'r') as f:
        json_f = json.load(f)
        camera_pose = np.array(json_f)
    print(camera_pose)

    i = 0
    out_bodys = []
    for dir_name in sorted(os.listdir(args.proxd_dir)):
        fullpath = os.path.join(args.proxd_dir, dir_name, '000.pkl')
        print(fullpath)
        output = bm.get_mesh_bypath(fullpath)
        out_bodys.append(output)
        i = i+1
        if i==80:
            break



    for i in range(80):
        scence = pyrender.Scene()

        env_mesh1 = pyrender.Mesh.from_trimesh(env_mesh)
        scence.add(env_mesh1)

        output1 = pyrender.Mesh.from_trimesh(out_bodys[i])
        scence.add(output1, pose=camera_pose)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

        camera_pose1 = np.array([
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3],
            [0.0, 0.0, 0.0, 1.0],
            ])
        scence.add(camera, pose=camera_pose1)
        light = pyrender.DirectionalLight(color=[1.0,1.0,1.0,1.0], intensity=5)

        scence.add(light, pose=camera_pose1)

        r = pyrender.OffscreenRenderer(1920, 1080)
        color, depth = r.render(scence)
        plt.figure()
        plt.imshow(color)
        plt.show()
        print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', default='/home/kaiwang/Documents/MpgModel', type=str, help='')
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')
    parser.add_argument('--num_pca_comps', type=int, default=12, help='')
    parser.add_argument('--scence_path', default='../datasets/PROX/scenes/BasementSittingBooth.ply', type=str, help='')
    parser.add_argument('--proxd_dir',
                        default='/home/kaiwang/Documents/DataSet/PROX/PROXD/BasementSittingBooth_00142_01/results',
                        type=str, help='')

    args = parser.parse_args()
    main(args)