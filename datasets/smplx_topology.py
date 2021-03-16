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
    [0, 1, torch.Tensor([0., 0., 0.])],
    [1, 4, torch.Tensor([0., 0., 0.])],
    [4, 7, torch.Tensor([0., 0., 0.])],
    [7, 10, torch.Tensor([0., 0., 0.])],
    [0, 2, torch.Tensor([0., 0., 0.])],
    [2, 5, torch.Tensor([0., 0., 0.])],
    [5, 8, torch.Tensor([0., 0., 0.])],
    [8, 11, torch.Tensor([0., 0., 0.])],
    [0, 3, torch.Tensor([0., 0., 0.])],
    [3, 6, torch.Tensor([0., 0., 0.])],
    [6, 9, torch.Tensor([0., 0., 0.])],
    [9, 12, torch.Tensor([0., 0., 0.])],
    [12, 15, torch.Tensor([0., 0., 0.])],
    [9, 13, torch.Tensor([0., 0., 0.])],
    [13, 16, torch.Tensor([0., 0., 0.])],
    [16, 18, torch.Tensor([0., 0., 0.])],
    [18, 20, torch.Tensor([0., 0., 0.])],
    [9, 14, torch.Tensor([0., 0., 0.])],
    [14, 17, torch.Tensor([0., 0., 0.])],
    [17, 19, torch.Tensor([0., 0., 0.])],
    [19, 21, torch.Tensor([0., 0., 0.])],
]

JOINT_NAMES_order = [
    'pelvis',

    'left_hip',
    'left_knee',
    'left_ankle',
    'left_foot',

    'right_hip',
    'right_knee',
    'right_ankle',
    'right_foot',

    'spine1',
    'spine2',
    'spine3',
    'neck',
    'head',

    'left_collar',
    'left_shoulder',
    'left_elbow',
    'left_wrist',

    'right_collar',
    'right_shoulder',
    'right_elbow',
    'right_wrist'
]

EDGES_order = [
    [0, 1, torch.Tensor([0., 0., 0.])],
    [1, 2, torch.Tensor([0., 0., 0.])],
    [2, 3, torch.Tensor([0., 0., 0.])],
    [3, 4, torch.Tensor([0., 0., 0.])],

    [0, 5, torch.Tensor([0., 0., 0.])],
    [5, 6, torch.Tensor([0., 0., 0.])],
    [6, 7, torch.Tensor([0., 0., 0.])],
    [7, 8, torch.Tensor([0., 0., 0.])],

    [0, 9, torch.Tensor([0., 0., 0.])],
    [9, 10, torch.Tensor([0., 0., 0.])],
    [10, 11, torch.Tensor([0., 0., 0.])],
    [11, 12, torch.Tensor([0., 0., 0.])],
    [12, 13, torch.Tensor([0., 0., 0.])],

    [11, 14, torch.Tensor([0., 0., 0.])],
    [14, 15, torch.Tensor([0., 0., 0.])],
    [15, 16, torch.Tensor([0., 0., 0.])],
    [16, 17, torch.Tensor([0., 0., 0.])],

    [11, 18, torch.Tensor([0., 0., 0.])],
    [18, 19, torch.Tensor([0., 0., 0.])],
    [19, 20, torch.Tensor([0., 0., 0.])],
    [20, 21, torch.Tensor([0., 0., 0.])],
]

MAP = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]

MAP_edge = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]