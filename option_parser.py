import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--proxd_dir', type=str, default='datasets/PROX/PROXD', help='PROXD Dataset path')
    parser.add_argument('--amass_dir', type=str, default='datasets/AMASS', help='PROXD Dataset path')
    parser.add_argument('--run_dir', type=str, default='run', help='run_dir')

    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--epoch_num', type=int, default=10000, help='epoch_num')
    parser.add_argument('--epoch_begin', type=int, default=0)
    parser.add_argument('--num_worker', type=int, default=16, help='num_worker')
    parser.add_argument('--windows_size', type=int, default=128, help='windows_size')
    parser.add_argument('--save_dir', type=str, default='run', help='PROXD Dataset path')
    parser.add_argument('--kernel_size', type=int, default=7, help='kernel_size')
    parser.add_argument('--num_layers', type=int, default=3, help='num_layers')
    parser.add_argument('--skeleton_dist', type=int, default=2, help='skeleton_dist')
    parser.add_argument('--padding_mode', type=str, default='reflect', help='padding_mode')
    parser.add_argument('--skeleton_pool', type=str, default='mean')
    parser.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    parser.add_argument('--standard_framerate', type=float, default=30.0, help='standard_framerate')
    parser.add_argument('--rotation', type=str, default='quaternion', help='rotation')
    parser.add_argument('--normalization', type=int, default=0, help='normalization')
    parser.add_argument('--fk_world', type=int, default=0)
    parser.add_argument('--pos_repr', type=str, default='3d')

    parser.add_argument('--model_folder', default='/home/kaiwang/Documents/MpgModel', type=str, help='')
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')
    parser.add_argument('--num_pca_comps', type=int, default=12, help='')
    parser.add_argument('--patch_gan', type=bool, default=True, help='')
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--gan_mode', type=str, default='lsgan')

    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_cyc', type=float, default=3)


    return parser

def get_args():
    parser = get_parser()
    return parser.parse_args()

def get_std_bvh(args=None, dataset=None):
    if args is None and dataset is None: raise Exception('Unexpected parameter')
    if dataset is None: dataset = args.dataset
    std_bvh = './datasets/Mixamo/std_bvhs/{}.bvh'.format(dataset)
    return std_bvh

def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))