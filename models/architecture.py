import torch
import smplx
from torch import nn
from torch import optim
from models.BaseModel import BaseModel
from models.skeleton_operator import find_neighbor,SkeletonConv,SkeletonPool,SkeletonUnpool
from human_body_prior.body_model.body_model import BodyModel

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

    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        return list(self.auto_encoder.parameters())

    def D_parameters(self):
        return list()

class GAN_model(BaseModel):
    def __init__(self, args):
        super(GAN_model, self).__init__(args)
        import models.smplx_topology
        self.args = args
        self.models = []
        self.D_para = []
        self.G_para = []

        bm_path = '/home/kaiwang/Documents/MpgModel/MANO/smplh/male/model.npz'
        dmpl_path = '/home/kaiwang/Documents/MpgModel/MANO/dmpls/male/model.npz'
        self.smplh = BodyModel(bm_path=bm_path, num_betas = 10, num_dmpls = 8,
                               batch_size= self.args.batch_size * self.args.windows_size,
                               path_dmpl = dmpl_path).to(args.cuda_device)

        model = IntegratedModel(args, models.smplx_topology.EDGES)
        self.models.append(model)
        self.D_para += model.D_parameters()
        self.G_para += model.G_parameters()

        #self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
        self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
        self.optimizers = [self.optimizerG]
        self.criterion_rec = torch.nn.MSELoss().to(self.device)

    def set_input(self, inputs):
        self.motions_input = inputs[0].to(self.args.cuda_device)
        self.betas_input = inputs[1].to(self.args.cuda_device)

    def compute_joints_pos(self, motion, betas):
        motions_tem = motion.permute(0, 2, 1).reshape(-1, 66)
        betas_tem = betas.permute(0, 2, 1).reshape(-1, 10)
        body = self.smplh(pose_body = motions_tem[:, 0:63], root_orient = motions_tem[:, 63:66], betas = betas_tem)
        joints = body.Jtr[:, 0:22, 0:3].reshape(self.args.batch_size, -1, self.args.windows_size)

        return joints

    def forward(self):
        self.motions = []
        self.latents = []
        self.res = []
        self.org_joints = []
        self.res_joints = []


        motion = self.motions_input
        betas = self.betas_input
        self.motions.append(motion)

        latent, res = self.models[0].auto_encoder(motion)
        self.latents.append(latent)
        self.res.append(res)

        org_joint = self.compute_joints_pos(motion, betas)
        self.org_joints.append(org_joint)
        res_joint = self.compute_joints_pos(res, betas)
        self.res_joints.append(res_joint)

    def backward_G(self):
        self.rec_loss = torch.zeros(1)
        self.rec_loss = self.rec_loss.requires_grad_()
        self.rec_loss = self.rec_loss.to(self.device)


        rec_loss1 = self.criterion_rec(self.motions[0], self.res[0])
        rec_loss2 = self.criterion_rec(self.org_joints[0], self.res_joints[0])
        rec_loss = rec_loss1 + rec_loss2
        self.rec_loss = self.rec_loss + rec_loss

        self.loss_G_total = self.rec_loss
        print(self.loss_G_total)
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()


