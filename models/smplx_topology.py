import smplx
import torch

JOINT_NUM = 22
EDGE_NUM = 21

JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist'
]

EDGES = [
    (0, 1, torch.Tensor([0., 0., 0.])),
    (1, 4, torch.Tensor([0., 0., 0.])),
    (4, 7, torch.Tensor([0., 0., 0.])),
    (7, 10, torch.Tensor([0., 0., 0.])),
    (0, 2, torch.Tensor([0., 0., 0.])),
    (2, 5, torch.Tensor([0., 0., 0.])),
    (5, 8, torch.Tensor([0., 0., 0.])),
    (8, 11, torch.Tensor([0., 0., 0.])),
    (0, 3, torch.Tensor([0., 0., 0.])),
    (3, 6, torch.Tensor([0., 0., 0.])),
    (6, 9, torch.Tensor([0., 0., 0.])),
    (9, 12, torch.Tensor([0., 0., 0.])),
    (12, 15, torch.Tensor([0., 0., 0.])),
    (9, 13, torch.Tensor([0., 0., 0.])),
    (13, 16, torch.Tensor([0., 0., 0.])),
    (16, 18, torch.Tensor([0., 0., 0.])),
    (18, 20, torch.Tensor([0., 0., 0.])),
    (9, 14, torch.Tensor([0., 0., 0.])),
    (14, 17, torch.Tensor([0., 0., 0.])),
    (17, 19, torch.Tensor([0., 0., 0.])),
    (19, 21, torch.Tensor([0., 0., 0.])),
]