# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List
from typing import Union

import sys
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp

__rank, __local_rank, __world_size, __device = 0, 0, 1, 'cpu'
__initialized = False


def initialized():
    return __initialized


def initialize(backend='nccl'):
    '''初始化distributed模式
    
    Args:
        backend (str): 模式名称
    '''
    global __device
    if not torch.cuda.is_available():
        print(f'[dist initialize] cuda is not available, use cpu instead', file=sys.stderr)
        return
    elif 'RANK' not in os.environ:
        __device = torch.empty(1).cuda().device
        print(f'[dist initialize] RANK is not set, use 1 GPU instead', file=sys.stderr)
        return
    
    # ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py#L29
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    global_rank, num_gpus = int(os.environ['RANK']), torch.cuda.device_count()
    local_rank = global_rank % num_gpus
    torch.cuda.set_device(local_rank)
    tdist.init_process_group(backend=backend)
    
    global __rank, __local_rank, __world_size, __initialized
    __local_rank = local_rank
    __rank, __world_size = tdist.get_rank(), tdist.get_world_size()
    __device = torch.empty(1).cuda().device
    __initialized = True
    
    assert tdist.is_initialized(), 'torch.distributed is not initialized!'


def get_rank():
    return __rank


def get_local_rank():
    return __local_rank


def get_world_size():
    return __world_size


def get_device():
    return __device


def is_master():
    return __rank == 0


def is_local_master():
    return __local_rank == 0


def barrier():
    if __initialized:
        tdist.barrier()


def parallelize(net, syncbn=False):
    '''将网络转换为并行化的网络'''
    if syncbn:
        # 将网络转换为同步BN
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # 将网络转换为GPU
    net = net.cuda()
    # 将网络添加到分布式数据平台中
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    return net


def allreduce(t: torch.Tensor) -> None:
    '''
    求t的所有元素并发送到所有给定的设备上
    '''
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.all_reduce(cu)
            t.copy_(cu.cpu())
        else:
            tdist.all_reduce(t)


def allgather(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    '''
    将t发送到所有组的rank，并将结果收集到ls中
    '''
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()
        ls = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls, t)
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def broadcast(t: torch.Tensor, src_rank) -> None:
    '''
    将tensor发送到指定的rank
    :param t: tensor
    :param src_rank: rank
    :return: None
    '''
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.broadcast(cu, src=src_rank)
            t.copy_(cu.cpu())
        else:
            tdist.broadcast(t, src=src_rank)
