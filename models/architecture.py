import torch
import smplx
from torch import nn
from torch import optim
from models.BaseModel import BaseModel
from models.Kinematics import ForwardKinematics
from models.skeleton_operator import find_neighbor,SkeletonConv,SkeletonPool,SkeletonUnpool,SkeletonLinear
from models.utils import GAN_loss, ImagePool
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args, topology, typeofchannel = 3):
        super(Encoder, self).__init__()
        self.topologies = [topology]

        self.channel_base = [typeofchannel]

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
                                    padding=self.padding, padding_mode=args.padding_mode, add_offset=True,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            if i < args.num_layers - 1: seq.append(nn.BatchNorm1d(out_channels))

            last_pool = True if i == args.num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.pooling_list[-1]))
            if i == args.num_layers - 1:
                self.last_channel = len(self.pooling_list[-1]) * self.channel_base[i + 1]
                '''
                seq = list()
                seq.append(SkeletonConv(neighbor_list, in_channels=self.last_channel, out_channels=128,
                                        joint_num=self.edge_num[i], kernel_size=1, stride=1))
                seq.append(nn.LeakyReLU(negative_slope=0.2))
                self.layers.append(nn.Sequential(*seq))
                '''


    def forward(self, input, offset=None):
        if self.channel_base[0] == 4:
            input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[i])
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

            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)

            '''
            if i == 0:
                seq = list()
                seq.append(SkeletonConv(neighbor_list, in_channels=128, out_channels=enc.last_channel,
                                        joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=1, stride=1))
                seq.append(nn.LeakyReLU(negative_slope=0.2))
                self.layers.append(nn.Sequential(*seq))
            '''

            seq = []
            self.unpools.append(
                SkeletonUnpool(enc.pooling_list[args.num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            seq.append(self.unpools[-1])
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode=args.padding_mode, add_offset=True,
                                    in_offset_channel=3 * enc.channel_base[args.num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        if self.enc.channel_base[0] == 4:
            input = input[:, :-1, :]
        return input

class Discriminator(nn.Module):
    def __init__(self, args, topology, typeofchannel = 3):
        super(Discriminator, self).__init__()
        self.topologies = [topology]
        self.channel_base = [typeofchannel]
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
                                    padding_mode=args.padding_mode, add_offset=False))
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
        return torch.sigmoid(input).squeeze()

class AE(nn.Module):
    def __init__(self, args, topology, typeofchannel = 3):
        super(AE, self).__init__()
        self.enc = Encoder(args, topology, typeofchannel)
        self.dec = Decoder(args, self.enc)

    def forward(self, input, offset=None):
        latent = self.enc(input, offset)
        result = self.dec(latent, offset)
        return latent, result


# eoncoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, args, edges):
        super(StaticEncoder, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(args.num_layers):
            neighbor_list = find_neighbor(edges, args.skeleton_dist)
            seq = []
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < args.num_layers - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels*2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
            seq.append(activation)
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input: torch.Tensor):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze(-1))
        return output

class IntegratedModel:
    # origin_offsets should have shape num_skeleton * J * 3
    def __init__(self, args, edges, typeofdata = 'SMPL'):
        self.args = args
        self.edges = edges

        if typeofdata == 'SMPL':
            self.auto_encoder = AE(args, topology=self.edges, typeofchannel = 4).to(args.cuda_device)
            self.static_encoder = StaticEncoder(args, edges = self.edges).to(args.cuda_device)
            self.discriminator = Discriminator(args, topology=self.edges, typeofchannel = 3).to(args.cuda_device)
            self.fk = ForwardKinematics(args, self.edges)
        elif typeofdata == 'MIXAMO':
            self.auto_encoder = AE(args, topology=self.edges, typeofchannel = 4).to(args.cuda_device)
            self.static_encoder = StaticEncoder(args, edges = self.edges).to(args.cuda_device)
            self.discriminator = Discriminator(args, topology = self.edges, typeofchannel = 3).to(args.cuda_device)
            self.fk = ForwardKinematics(args, self.edges)

    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters())

    def D_parameters(self):
        return list(self.discriminator.parameters())

    def save(self, epoch, ith_modle):
        import os
        run_path = self.args.run_dir
        path = os.path.join(run_path, 'model', str(epoch))
        if not os.path.exists(path):
            os.system('mkdir -p {}'.format(path))

        torch.save(self.auto_encoder.state_dict(), os.path.join(path, 'auto_encoder_'+str(ith_modle)+'.pt'))
        torch.save(self.static_encoder.state_dict(), os.path.join(path, 'static_encoder_'+str(ith_modle)+'.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator_'+str(ith_modle)+'.pt'))


    def load(self, epoch, ith_modle):
        import os
        run_path = self.args.run_dir
        path = os.path.join(run_path, 'model', str(epoch))

        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        self.auto_encoder.load_state_dict(torch.load(os.path.join(path, 'auto_encoder_'+str(ith_modle)+'.pt'),
                                                     map_location=self.args.cuda_device))
        self.static_encoder.load_state_dict(torch.load(os.path.join(path, 'static_encoder_'+str(ith_modle)+'.pt'),
                                                       map_location=self.args.cuda_device))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator_'+str(ith_modle)+'.pt'),
                                                      map_location=self.args.cuda_device))



class GAN_model(BaseModel):
    def __init__(self, args, dataset):
        super(GAN_model, self).__init__(args)
        self.args = args
        self.dataset = dataset

        import datasets.smplx_topology
        self.smpl_topy = datasets.smplx_topology.EDGES_order
        self.smpl_joints_num = datasets.smplx_topology.JOINT_NUM

        self.epoch_cnt = 0
        self.models = []
        self.D_para = []
        self.G_para = []

        #self.smplx = smplx.create(args.model_folder, model_type='smplx', batch_size = args.batch_size * args.windows_size).to(args.cuda_device)
        self.smplx_test = smplx.create(args.model_folder, model_type='smplx', batch_size = 1)

        #smpl
        model = IntegratedModel(args, self.smpl_topy)
        self.models.append(model)
        self.G_para += model.G_parameters()
        self.D_para += model.D_parameters()

        #mixamo
        model = IntegratedModel(args, dataset.mixamo.edges, typeofdata = 'MIXAMO')
        self.models.append(model)
        self.G_para += model.G_parameters()
        self.D_para += model.D_parameters()

        self.fake_pools = []
        self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
        self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
        self.optimizers = [self.optimizerG, self.optimizerD]

        self.criterion_rec = torch.nn.MSELoss().to(self.device)
        self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)
        self.criterion_cycle = torch.nn.L1Loss().to(self.device)

        self.fake_pools.append(ImagePool(args.pool_size))
        self.fake_pools.append(ImagePool(args.pool_size))

        self.writers = []
        from datasets.bvh_writer import BVH_writer
        from datasets.bvh_parser import BVH_file
        import option_parser
        file = BVH_file(option_parser.get_std_bvh(dataset='JEAN'))
        writer = BVH_writer(file.edges, file.names)
        self.writers.append(writer)

        smpl_topy = datasets.smplx_topology.EDGES_order
        MAP = datasets.smplx_topology.MAP
        def create_smpl_topy():
            statics_tem = torch.zeros(16).reshape(-1, 16)
            body = self.smplx_test(betas = statics_tem[:, 0:10])
            joints = body.joints[0, 0:22, 0:3]

            static_res = torch.zeros_like(joints)

            for i in range(len(smpl_topy)):
                edge = smpl_topy[i]
                static_res[i] = joints[MAP[edge[1]]] - joints[MAP[edge[0]]]
            return static_res.reshape(-1, 3)

        offsets = create_smpl_topy()
        for i in range(len(smpl_topy)):
            smpl_topy[i][2] = offsets[i]

        self.amass_offset = offsets

        self.dataset.amass.offset = self.amass_offset

        self.amass_offset = self.dataset.amass.offset.to(self.device).repeat(self.args.batch_size, 1, 1)
        self.mixamo_offset = self.dataset.mixamo.offset.to(self.device).repeat(self.args.batch_size, 1, 1)

        writer = BVH_writer(smpl_topy, datasets.smplx_topology.JOINT_NAMES_order)
        self.writers.append(writer)

    def set_input(self, inputs):
        self.amass_dynamic = inputs[0][0].to(self.args.cuda_device)
        self.amass_static = inputs[0][1].to(self.args.cuda_device)
        self.mixamo_dynamic = inputs[1].to(self.args.cuda_device)

    def compute_joints_pos(self, smplx_model, poses, statics):
        poses_tem = poses.permute(0, 2, 1).reshape(-1, 66)
        statics_tem = statics.permute(0, 2, 1).reshape(-1, 16)

        body = smplx_model(body_pose = poses_tem[:, 0:63],
                          betas=statics_tem[:, 0:10],
                          #global_orient = statics_tem[:, 10:13],
                          transl = poses_tem[:, 63:66])

        joints = body.joints[:, 0:22, 0:3].reshape(self.args.batch_size, -1, self.args.windows_size)

        return joints

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad

    def forward(self, with_noise = True):

        self.offset_reprs = []

        self.poses = []
        self.statics = []

        self.latent_poses = []
        self.res_poses = []

        self.res_statics = []

        self.fake_res = []
        self.fake_latent = []

        #amass
        offset_repr = self.models[0].static_encoder(self.amass_offset)
        self.offset_reprs.append(offset_repr)

        pose = self.amass_dynamic
        self.poses.append(pose)
        if with_noise:
            noise = torch.normal(std = 0.15, mean = 0.0, size = pose.shape) / 10.0
            noise = noise.to(self.args.cuda_device)
            pose = self.amass_dynamic + noise

        latent_pose, res_pose = self.models[0].auto_encoder(pose, offset_repr)
        self.latent_poses.append(latent_pose)
        self.res_poses.append(res_pose)


        #mixamo
        offset_repr = self.models[1].static_encoder(self.mixamo_offset)
        self.offset_reprs.append(offset_repr)

        pose = self.mixamo_dynamic
        self.poses.append(pose)
        if with_noise:
            noise = torch.normal(std=0.15, mean=0.0, size=pose.shape) / 10.0
            noise = noise.to(self.args.cuda_device)
            pose = self.mixamo_dynamic + noise

        latent_pose, res_pose = self.models[1].auto_encoder(pose, offset_repr)
        self.latent_poses.append(latent_pose)
        self.res_poses.append(res_pose)


        #amass -> amass Debug offset
        fake_res = self.models[0].auto_encoder.dec(self.latent_poses[0], self.offset_reprs[0])
        fake_latent = self.models[0].auto_encoder.enc(fake_res, self.offset_reprs[0])

        self.fake_res.append(fake_res)
        self.fake_latent.append(fake_latent)

        # amass -> mixamo
        fake_res = self.models[1].auto_encoder.dec(self.latent_poses[0], self.offset_reprs[1])
        fake_latent = self.models[1].auto_encoder.enc(fake_res, self.offset_reprs[1])

        self.fake_res.append(fake_res)
        self.fake_latent.append(fake_latent)

        # mixamo -> amass
        fake_res = self.models[0].auto_encoder.dec(self.latent_poses[1], self.offset_reprs[0])
        fake_latent = self.models[0].auto_encoder.enc(fake_res, self.offset_reprs[0])

        self.fake_res.append(fake_res)
        self.fake_latent.append(fake_latent)

        # mixamo -> mixamo
        fake_res = self.models[1].auto_encoder.dec(self.latent_poses[1], self.offset_reprs[1])
        fake_latent = self.models[1].auto_encoder.enc(fake_res, self.offset_reprs[1])

        self.fake_res.append(fake_res)
        self.fake_latent.append(fake_latent)




    def backward_G(self):

        self.org_positions = []
        self.res_positions = []

        self.fake_positions = []


        # amass 0
        org_position = self.models[0].fk.forward_from_raw(self.poses[0],
                                                          self.amass_offset.reshape(self.args.batch_size,-1, 3))
        org_position = org_position.reshape(self.args.batch_size, self.args.windows_size, -1).permute(0, 2, 1)
        self.org_positions.append(org_position)

        res_position = self.models[0].fk.forward_from_raw(self.res_poses[0],
                                                          self.amass_offset.reshape(self.args.batch_size,-1, 3))
        res_position = res_position.reshape(self.args.batch_size, self.args.windows_size, -1).permute(0, 2, 1)
        self.res_positions.append(res_position)


        # mixamo 1
        org_position = self.models[1].fk.forward_from_raw(self.poses[1],
                                                          self.mixamo_offset.reshape(self.args.batch_size,-1, 3))
        org_position = org_position.reshape(self.args.batch_size, self.args.windows_size, -1).permute(0, 2, 1)
        self.org_positions.append(org_position)

        res_position = self.models[1].fk.forward_from_raw(self.res_poses[1],
                                                          self.mixamo_offset.reshape(self.args.batch_size,-1, 3))
        res_position = res_position.reshape(self.args.batch_size, self.args.windows_size, -1).permute(0, 2, 1)
        self.res_positions.append(res_position)


        # fake amass
        fake_position = self.models[0].fk.forward_from_raw(self.fake_res[2],
                                                          self.amass_offset.reshape(self.args.batch_size,-1, 3))
        fake_position = fake_position.reshape(self.args.batch_size, self.args.windows_size, -1).permute(0, 2, 1)
        self.fake_positions.append(fake_position)

        # fake mixamo
        fake_position = self.models[1].fk.forward_from_raw(self.fake_res[1],
                                                           self.mixamo_offset.reshape(self.args.batch_size, -1, 3))
        fake_position = fake_position.reshape(self.args.batch_size, self.args.windows_size, -1).permute(0, 2, 1)
        self.fake_positions.append(fake_position)

        # rec Loss
        self.rec_loss_total = torch.zeros(1)
        self.rec_loss_total = self.rec_loss_total.requires_grad_()
        self.rec_loss_total = self.rec_loss_total.to(self.device)


        rec_loss1_1 = self.criterion_rec(self.poses[0], self.res_poses[0])
        rec_loss1_2 = self.criterion_rec(self.org_positions[0], self.res_positions[0])

        rec_loss2_1 = self.criterion_rec(self.poses[1], self.res_poses[1])
        rec_loss2_2 = self.criterion_rec(self.org_positions[1], self.res_positions[1])
        

        self.rec_loss_total = self.rec_loss_total + \
                              rec_loss1_1 * 50.0 + rec_loss1_2 * 50.0 +\
                              rec_loss2_1 * 50.0 + rec_loss2_2 * 50.0



        # cyc Loss
        self.cycle_loss = torch.zeros(1)
        self.cycle_loss = self.cycle_loss.requires_grad_()
        self.cycle_loss = self.cycle_loss.to(self.device)

        cycle_loss_1_1 = self.criterion_cycle(self.latent_poses[0], self.fake_latent[0])
        cycle_loss_1_2 = self.criterion_cycle(self.latent_poses[0], self.fake_latent[1])
        cycle_loss_2_1 = self.criterion_cycle(self.latent_poses[1], self.fake_latent[2])
        cycle_loss_2_2 = self.criterion_cycle(self.latent_poses[1], self.fake_latent[3])

        self.cycle_loss = cycle_loss_1_1 + cycle_loss_1_2 + cycle_loss_2_1 + cycle_loss_2_2

        # GAN Loss
        self.loss_G = torch.zeros(1)
        self.loss_G = self.loss_G.requires_grad_()
        self.loss_G = self.loss_G.to(self.device)

        loss_G_1 = self.criterion_gan(self.models[0].discriminator(self.fake_positions[0]), True)
        loss_G_2 = self.criterion_gan(self.models[1].discriminator(self.fake_positions[1]), True)

        self.loss_G = self.loss_G + loss_G_1 + loss_G_2

        self.loss_G_total = self.rec_loss_total * self.args.lambda_rec + self.cycle_loss * self.args.lambda_cyc + self.loss_G * 5

        self.loss_G_total.backward(retain_graph=True)

        self.loss_recoder.add_scalar("rec_loss1_1", rec_loss1_1)
        self.loss_recoder.add_scalar("rec_loss1_2", rec_loss1_2)
        self.loss_recoder.add_scalar("rec_loss2_1", rec_loss2_1)
        self.loss_recoder.add_scalar("rec_loss2_2", rec_loss2_2)
        self.loss_recoder.add_scalar("rec_loss_total", self.rec_loss_total)


        self.loss_recoder.add_scalar("loss_G_1", loss_G_1)
        self.loss_recoder.add_scalar("loss_G_2", loss_G_2)
        self.loss_recoder.add_scalar("loss_G", self.loss_G)

        self.loss_recoder.add_scalar("cycle_loss_1_1", cycle_loss_1_1)
        self.loss_recoder.add_scalar("cycle_loss_1_2", cycle_loss_1_2)
        self.loss_recoder.add_scalar("cycle_loss_2_1", cycle_loss_2_1)
        self.loss_recoder.add_scalar("cycle_loss_2_2", cycle_loss_2_2)
        self.loss_recoder.add_scalar("cycle_loss", self.cycle_loss)


        self.loss_recoder.add_scalar("loss_G_total", self.loss_G_total)


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

        fake = self.fake_pools[0].query(self.fake_positions[0])
        self.loss_Ds.append(self.backward_D_basic(self.models[0].discriminator, self.org_positions[0].detach(), fake))

        fake = self.fake_pools[1].query(self.fake_positions[1])
        self.loss_Ds.append(self.backward_D_basic(self.models[1].discriminator, self.org_positions[1].detach(), fake))


        self.loss_D = self.loss_D + self.loss_Ds[0] + self.loss_Ds[1]

        self.loss_recoder.add_scalar("loss_D_1", self.loss_Ds[0])
        self.loss_recoder.add_scalar("loss_D_2", self.loss_Ds[1])
        self.loss_recoder.add_scalar("loss_D", self.loss_D)

    def optimize_parameters(self):
        self.forward(with_noise = False)

        self.discriminator_requires_grad_(False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.discriminator_requires_grad_(True)
        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()


    def save(self):
        import os
        for i, model in enumerate(self.models):
            model.save(self.epoch_cnt, i)

        optimizerG_name = os.path.join(self.args.save_dir, 'model/{}/optimizerG.pt'.format(self.epoch_cnt))
        torch.save(self.optimizerG.state_dict(), optimizerG_name)

        optimizerD_name = os.path.join(self.args.save_dir, 'model/{}/optimizerD.pt'.format(self.epoch_cnt))
        torch.save(self.optimizerD.state_dict(), optimizerD_name)

        print('Save succeed!')

    def load(self, epoch = 0):
        import os

        for i, model in enumerate(self.models):
            model.load(self.epoch_cnt, i)

        optimizerG_name = os.path.join(self.args.save_dir, 'model/{}/optimizerG.pt'.format(epoch))
        self.optimizerG.load_state_dict(torch.load(optimizerG_name))

        optimizerD_name = os.path.join(self.args.save_dir, 'model/{}/optimizerD.pt'.format(epoch))
        self.optimizerD.load_state_dict(torch.load(optimizerD_name))

        self.epoch_cnt = epoch

        print('Load succeed!')

    def test(self, scence_name, filename):
        self.forward(with_noise = False)
        self.test_vis(scence_name, filename, True)
        self.test_vis(scence_name, filename, False)


    def test_vis(self, scence_name, filename, use_org = False):
        import pyrender
        import trimesh
        import json
        import numpy as np
        import cv2

        env_mesh = trimesh.load('/home/kaiwang/Documents/DataSet/PROX/scenes/'+scence_name +'.ply')

        with open('/home/kaiwang/Documents/DataSet/PROX/cam2world/' + scence_name + '.json', 'r') as f:
            json_f = json.load(f)
            camera_pose = np.array(json_f)

        pose = self.poses[0].permute(0, 2, 1).squeeze()
        static = self.statics[0].permute(0, 2, 1).squeeze()
        length = pose.shape[0]

        res_pose = self.res_poses[0].permute(0, 2, 1).squeeze()
        res_static = self.res_statics[0].permute(0, 2, 1).squeeze()

        scence = pyrender.Scene()
        env_mesh1 = pyrender.Mesh.from_trimesh(env_mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        camera_pose1 = np.array([
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0, 1.0], intensity=2)

        video_reslou = (1280, 720)
        r = pyrender.OffscreenRenderer(viewport_width=video_reslou[0], viewport_height=video_reslou[1], point_size=1.0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if use_org:
            path = self.args.save_dir + '/video/' + filename + "_org_scence.mp4"
        else:
            path = self.args.save_dir + '/video/' + filename + "_res_scence.mp4"

        video_writer = cv2.VideoWriter(path, fourcc, 30, video_reslou)

        for item in range(length):
            scence.clear()

            scence.add(env_mesh1)
            scence.add(camera, pose=camera_pose1)
            scence.add(light, pose=camera_pose1)

            print(item)
            if use_org:
                body = self.smplx_test(body_pose=pose[item:item + 1, 0:63],
                                       betas=static[0:0 + 1, 0:10],
                                       global_orient=static[item:item + 1, 10:13],
                                       transl=static[item:item + 1, 13:16]
                                       )
            else:
                body = self.smplx_test(body_pose=res_pose[item:item + 1, 0:63],
                                       betas=static[0:0 + 1, 0:10],
                                       global_orient=static[item:item + 1, 10:13],
                                       transl=static[item:item + 1, 13:16]
                                       )

            vertices = body.vertices.detach().cpu().numpy().squeeze()

            body = trimesh.Trimesh(vertices, self.smplx_test.faces)
            body = pyrender.Mesh.from_trimesh(body)

            scence.add(body, pose=camera_pose)

            color, depth = r.render(scence)
            color = color[:,:,[2, 1, 0]]
            color = color.astype(np.uint8)
            video_writer.write(color)

        video_writer.release()

    def compute_test_result(self, prox_data):
        import os

        self.writers[1].write_raw(self.poses[0][0], 'quaternion', os.path.join('run/', 'org_amass.bvh'))
        self.writers[1].write_raw(self.res_poses[0][0], 'quaternion', os.path.join('run/', 'res_amass.bvh'))
        self.writers[0].write_raw(self.poses[1][0], 'quaternion', os.path.join('run/', 'org_mixamo.bvh'))
        self.writers[0].write_raw(self.res_poses[1][0], 'quaternion', os.path.join('run/', 'res_mixamo.bvh'))
        self.writers[0].write_raw(self.fake_res[1][0], 'quaternion', os.path.join('run/', 'fake_amass.bvh'))

        '''
        self.amass_dynamic = prox_data[0].to(self.args.cuda_device)
        self.amass_static = prox_data[1].to(self.args.cuda_device)
        
        self.forward(with_noise = False)




        pose1 = self.poses[0][0]
        pose2 = self.res_poses[0][0]
        pose3 = self.fake_res[1][0]
        self.writers[1].write_raw(pose1, 'quaternion', os.path.join('run/', 'org_prox.bvh'))
        self.writers[1].write_raw(pose2, 'quaternion', os.path.join('run/', 'res_prox.bvh'))
        self.writers[0].write_raw(pose3, 'quaternion', os.path.join('run/', 'fake_prox.bvh'))
        '''











