import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--proxd_dir', type=str, default='datasets/PROX/PROXD', help='PROXD Dataset path')
    parser.add_argument('--amass_dir', type=str, default='datasets/AMASS', help='PROXD Dataset path')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--epoch_num', type=int, default=10000, help='epoch_num')
    parser.add_argument('--epoch_begin', type=int, default=0)
    parser.add_argument('--num_worker', type=int, default=16, help='num_worker')
    parser.add_argument('--windows_size', type=int, default=64, help='windows_size')



    return parser

def get_args():
    parser = get_parser()
    return parser.parse_args()