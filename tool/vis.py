import open3d as o3d
import numpy as np
import argparse
import pickle
import smplx
import torch
import os

class BodyMaker:
    def __init__(self, args):
        self.model = smplx.create(args.model_folder, model_type='smplx',
                             gender=args.gender,
                             num_pca_comps=args.num_pca_comps
                             )
        self.body = o3d.geometry.TriangleMesh()

    def get_mesh_bypath(self, path, trans):
        with open(path, 'rb') as f:
            param = pickle.load(f, encoding='latin1')

        torch_param = {}
        for key in param.keys():
            torch_param[key] = torch.tensor(param[key])

        #torch_param['transl'] = torch.zeros(torch_param['transl'].shape)
        #torch_param['global_orient'] = torch.zeros(torch_param['global_orient'].shape)
        #torch_param['betas'] = torch.zeros(torch_param['betas'].shape)
        #torch_param['body_pose'] = torch.zeros(torch_param['body_pose'].shape)


        output = self.model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        self.body.vertices = o3d.utility.Vector3dVector(vertices)
        self.body.triangles = o3d.utility.Vector3iVector(self.model.faces)
        self.body.vertex_normals = o3d.utility.Vector3dVector([])
        self.body.triangle_normals = o3d.utility.Vector3dVector([])
        self.body.compute_vertex_normals()
        self.body.transform(trans)
        return self.body


class StaticScence:
    def __init__(self, args):
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0.0, 0.0, 0.0])
        self.mesh = o3d.io.read_triangle_mesh(args.scence_path)

        import json
        with open("/home/kaiwang/Documents/DataSet/PROX/cam2world/BasementSittingBooth.json", 'r') as f:
            self.trans = np.array(json.load(f))

    def get_mesh(self):
        return self.mesh_frame + self.mesh

def main(args):

    bodymaker = BodyMaker(args)
    staticscence = StaticScence(args)

    output = dict()
    output['scence'] = staticscence.get_mesh()

    for dir_name in sorted(os.listdir(args.proxd_dir)):
        fullpath = os.path.join(args.proxd_dir, dir_name, '000.pkl')
        print(fullpath)
        output['body'] = bodymaker.get_mesh_bypath(fullpath, staticscence.trans)
        o3d.visualization.draw(list(output.values()))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', default='/home/kaiwang/Documents/MpgModel', type=str, help='')
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')
    parser.add_argument('--num_pca_comps', type=int, default=12, help='')
    parser.add_argument('--scence_path', default='../datasets/PROX/scenes/BasementSittingBooth.ply', type=str, help='')
    parser.add_argument('--proxd_dir', default='/home/kaiwang/Documents/DataSet/PROX/PROXD/BasementSittingBooth_00142_01/results', type=str, help='')

    args = parser.parse_args()
    main(args)