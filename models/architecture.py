import torch
import smplx
from torch import nn
from torch import optim
from models.BaseModel import BaseModel
from models.skeleton_operator import find_neighbor,SkeletonConv,SkeletonPool,SkeletonUnpool
from models.utils import GAN_loss, ImagePool

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
        self.channel_base = [3]

        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        self.kernel_size = args.kernel_size
        self.padding = (self.kernel_size - 1) // 2

        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = list()
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=self.kernel_size, stride=2,
                                    padding=self.padding, padding_mode=args.padding_mode))
            last_pool = True if i == args.num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input):
        for i, layer in enumerate(self.layers):
            input = layer(input)
        return input


class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        for i in range(args.num_layers):
            seq = []
            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)


            self.unpools.append(
                SkeletonUnpool(enc.pooling_list[args.num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            seq.append(self.unpools[-1])
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode=args.padding_mode))
            self.convs.append(seq[-1])
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input):
        for i, layer in enumerate(self.layers):
            input = layer(input)

        return input

class Discriminator(nn.Module):
    def __init__(self, args, topology):
        super(Discriminator, self).__init__()
        self.topologies = [topology]
        self.channel_base = [3]
        self.channel_list = []
        self.joint_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.joint_num[i]
            out_channels = self.channel_base[i+1] * self.joint_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            if i == args.num_layers - 1:
                kernel_size = 16
                padding = 0

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.joint_num[i], kernel_size=kernel_size, stride=2, padding=padding,
                                    padding_mode=args.padding_mode))
            if i < args.num_layers - 1: seq.append(nn.BatchNorm1d(out_channels))
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list))
            seq.append(pool)
            if not self.args.patch_gan or i < args.num_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.joint_num.append(len(pool.new_edges) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.joint_num[-1] * self.channel_base[i+1]

        if not args.patch_gan: self.compress = nn.Linear(in_features=self.last_channel, out_features=1)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        if not self.args.patch_gan:
            input = input.reshape(input.shape[0], -1)
            input = self.compress(input)
        # shape = (64, 72, 9)
        return torch.sigmoid(input).squeeze()

class AE(nn.Module):
    def __init__(self, args, topology):
        super(AE, self).__init__()
        self.enc = Encoder(args, topology)
        self.dec = Decoder(args, self.enc)

    def forward(self, input):
        latent = self.enc(input)
        result = self.dec(latent)
        return latent, result


class IntegratedModel:
    # origin_offsets should have shape num_skeleton * J * 3
    def __init__(self, args, edges):
        self.args = args
        self.edges = edges
        self.auto_encoder = AE(args, topology=self.edges).to(args.cuda_device)
        self.discriminator = Discriminator(args, topology=self.edges).to(args.cuda_device)

    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        return list(self.auto_encoder.parameters())

    def D_parameters(self):
        return list(self.discriminator.parameters())

class GAN_model(BaseModel):
    def __init__(self, args):
        super(GAN_model, self).__init__(args)
        import models.smplx_topology
        self.args = args
        self.models = []
        self.D_para = []
        self.G_para = []

        self.smplx = smplx.create(args.model_folder, model_type='smplx', batch_size = args.batch_size * args.windows_size).to(args.cuda_device)
        self.smplx_test = smplx.create(args.model_folder, model_type='smplx', batch_size = 1).to(args.cuda_device)

        model = IntegratedModel(args, models.smplx_topology.EDGES)
        self.models.append(model)
        self.D_para += model.D_parameters()
        self.G_para += model.G_parameters()

        self.fake_pools = []
        self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
        self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
        self.optimizers = [self.optimizerG, self.optimizerD]

        self.criterion_rec = torch.nn.MSELoss().to(self.device)
        self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)

        self.fake_pools.append(ImagePool(args.pool_size))

    def set_input(self, inputs):
        self.motions_input = inputs[0].to(self.args.cuda_device)
        self.betas_input = inputs[1].to(self.args.cuda_device)
        self.root_trans = inputs[2].to(self.args.cuda_device)

    def compute_joints_pos(self, motion, betas, root_trans):
        motions_tem = motion.permute(0, 2, 1).reshape(-1, 66)
        betas_tem = betas.permute(0, 2, 1).reshape(-1, 10)
        root_trans_tem = root_trans.permute(0, 2, 1).reshape(-1, 3)


        body = self.smplx(body_pose = motions_tem[:, 0:63],
                        global_orient = motions_tem[:, 63:66],
                        betas = betas_tem[:, 0:10],
                        transl = root_trans_tem[:, 0:3])

        joints = body.joints[:, 0:22, 0:3].reshape(self.args.batch_size, -1, self.args.windows_size)

        return joints

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad

    def forward(self):
        self.motions = []
        self.latents = []
        self.res = []

        motion = self.motions_input
        self.motions.append(motion)
        latent, res = self.models[0].auto_encoder(motion)

        self.latents.append(latent)
        self.res.append(res)

    def backward_G(self):

        self.pos_ref = []
        self.res_pos = []

        org_joint = self.compute_joints_pos(self.motions[0], self.betas_input, self.root_trans)
        self.pos_ref.append(org_joint)
        res_joint = self.compute_joints_pos(self.res[0], self.betas_input, self.root_trans)
        self.res_pos.append(res_joint)

        self.rec_loss = torch.zeros(1)
        self.rec_loss = self.rec_loss.requires_grad_()
        self.rec_loss = self.rec_loss.to(self.device)


        rec_loss1 = self.criterion_rec(self.motions[0], self.res[0])
        rec_loss2 = self.criterion_rec(self.pos_ref[0], self.res_pos[0])
        rec_loss = rec_loss1 + rec_loss2
        self.rec_loss = self.rec_loss + rec_loss


        self.loss_G = self.criterion_gan(self.models[0].discriminator(self.res_pos[0]), True)

        self.loss_G_total = self.rec_loss * self.args.lambda_rec + \
                            self.loss_G
        self.loss_G_total.backward()

        self.loss_recoder.add_scalar("rec_loss_total", self.rec_loss)
        self.loss_recoder.add_scalar("rec_loss1", rec_loss1)
        self.loss_recoder.add_scalar("rec_loss2", rec_loss2)
        self.loss_recoder.add_scalar("loss_G", self.loss_G)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_Ds = []
        self.loss_D = 0

        fake = self.fake_pools[0].query(self.res_pos[0])
        self.loss_Ds.append(self.backward_D_basic(self.models[0].discriminator, self.pos_ref[0].detach(), fake))
        self.loss_D = self.loss_D + self.loss_Ds[-1]

        self.loss_recoder.add_scalar("loss_D", self.loss_D)

    def optimize_parameters(self):
        self.forward()

        self.discriminator_requires_grad_(False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.discriminator_requires_grad_(True)
        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()


    def test(self, scence_name):
        self.forward()
        self.test_vis(scence_name)
        print("test")

    def test_vis(self, scence_name):
        import pyrender
        import trimesh
        import json
        import skvideo.io
        import numpy as np

        env_mesh = trimesh.load('/home/kaiwang/Documents/DataSet/PROX/scenes/'+scence_name +'.ply')

        with open('/home/kaiwang/Documents/DataSet/PROX/cam2world/' + scence_name + '.json', 'r') as f:
            json_f = json.load(f)
            camera_pose = np.array(json_f)



        motion = self.motions[0].permute(0, 2, 1).squeeze()
        res = self.res[0].permute(0, 2, 1).squeeze()
        betas = self.betas_input.permute(0, 2, 1).squeeze()
        root_trans_tem = self.root_trans.permute(0, 2, 1).squeeze()

        length = res.shape[0]

        frame_vedio = []
        r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1.0)
        for item in range(700):
            scence = pyrender.Scene()
            env_mesh1 = pyrender.Mesh.from_trimesh(env_mesh)
            scence.add(env_mesh1)

            print(item)
            body = self.smplx_test(body_pose=res[item:item+1, 0:63],
                              global_orient=motion[item:item+1, 63:66],
                              betas=betas[item:item+1, 0:10],
                              transl=root_trans_tem[item:item+1, 0:3])
            vertices = body.vertices.detach().cpu().numpy().squeeze()

            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            body = trimesh.Trimesh(vertices, self.smplx_test.faces,
                                       vertex_colors=vertex_colors)
            body = pyrender.Mesh.from_trimesh(body)
            scence.add(body, pose=camera_pose)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
            camera_pose1 = np.array([
                [1.0, 0.0, 0.0, 0.3],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3],
                [0.0, 0.0, 0.0, 1.0],
            ])
            scence.add(camera, pose=camera_pose1)

            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0, 1.0], intensity=3)
            scence.add(light, pose=np.array([
                                [1.0, 0.0, 0.0, 0.3],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 3],
                                [0.0, 0.0, 0.0, 1.0],
                                ]))

            color, depth = r.render(scence)
            frame_vedio.append(color)

        frame_vedio = np.stack(frame_vedio, axis=0)
        outputdata = frame_vedio
        outputdata = outputdata.astype(np.uint8)
        print(outputdata.shape)

        skvideo.io.vwrite("outputvideo.mp4", outputdata)





