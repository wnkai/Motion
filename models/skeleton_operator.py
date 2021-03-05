import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SkeletonConv(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels,
                 kernel_size, joint_num, stride=1,
                 padding=0,padding_mode='zeros'):

        super(SkeletonConv, self).__init__()

        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num

        self.joint_num = joint_num
        self.neighbour_list = neighbour_list
        self.expanded_neighbour_list = []

        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        self.bias = torch.zeros(out_channels)
        self.mask = torch.zeros_like(self.weight)
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            self.mask[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...] = 1
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                       weight_masked, self.bias, self.stride,
                       0, self.dilation, self.groups)
        return res

class SkeletonLinear(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, extra_dim1=False):
        super(SkeletonLinear, self).__init__()
        self.neighbour_list = neighbour_list
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_joint = in_channels // len(neighbour_list)
        self.out_channels_per_joint = out_channels // len(neighbour_list)
        self.expanded_neighbour_list = []

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels)
        self.mask = torch.zeros(out_channels, in_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        weight_masked = self.weight * self.mask
        res = F.linear(input, weight_masked, self.bias)
        return res

class SkeletonPool(nn.Module):
    def __init__(self, edges, pooling_mode, channels_per_edge, last_pool=False):
        super(SkeletonPool, self).__init__()

        self.channels_per_edge = channels_per_edge
        self.pooling_mode = pooling_mode
        self.edge_num = len(edges) + 1
        self.seq_list = []
        self.pooling_list = []
        self.new_edges = []
        degree = [0] * 100

        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1

        def find_seq(j, seq):
            nonlocal self, degree, edges

            if degree[j] > 2 and j != 0:
                self.seq_list.append(seq)
                seq = []

            if degree[j] == 1:
                self.seq_list.append(seq)
                return

            for idx, edge in enumerate(edges):
                if edge[0] == j:
                    find_seq(edge[1], seq + [idx])

        find_seq(0, [])
        for seq in self.seq_list:
            if last_pool:
                self.pooling_list.append(seq)
                continue
            if len(seq) % 2 == 1:
                self.pooling_list.append([seq[0]])
                self.new_edges.append(edges[seq[0]])
                seq = seq[1:]
            for i in range(0, len(seq), 2):
                self.pooling_list.append([seq[i], seq[i + 1]])
                self.new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])

        # add global position
        self.pooling_list.append([self.edge_num - 1])

        self.weight = torch.zeros(len(self.pooling_list) * channels_per_edge, self.edge_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[i * channels_per_edge + c, j * channels_per_edge + c] = 1.0 / len(pair)

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input)


class SkeletonUnpool(nn.Module):
    def __init__(self, pooling_list, channels_per_edge):
        super(SkeletonUnpool, self).__init__()
        self.pooling_list = pooling_list
        self.input_edge_num = len(pooling_list)
        self.output_edge_num = 0
        self.channels_per_edge = channels_per_edge
        for t in self.pooling_list:
            self.output_edge_num += len(t)

        self.weight = torch.zeros(self.output_edge_num * channels_per_edge, self.input_edge_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[j * channels_per_edge + c, i * channels_per_edge + c] = 1

        self.weight = nn.Parameter(self.weight)
        self.weight.requires_grad_(False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input)