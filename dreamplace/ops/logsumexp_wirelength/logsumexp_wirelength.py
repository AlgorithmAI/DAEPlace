##
# @file   logsumexp_wirelength.py
# @author Xu Li
# @date   10 2024
# @brief  Compute log-sum-exp wirelength and gradient according to NTUPlace3
#

import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength_cpp as logsumexp_wirelength_cpp
try:
    import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength_hip as logsumexp_wirelength_hip
    import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength_hip_atomic as logsumexp_wirelength_hip_atomic
except:
    pass
import pdb

class LogSumExpWirelengthFunction(Function):
    """compute weighted average wirelength.
    @param pos pin location (x array, y array), not cell location
    @param flat_netpin flat netpin map, length of #pins
    @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins
    @param gamma the smaller, the closer to HPWL
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, netpin_values, net_mask, gamma, num_threads):
        if pos.is_cuda:
            output = logsumexp_wirelength_hip.forward(pos.view(pos.numel()), flat_netpin, netpin_start, netpin_values, net_mask, gamma)
        else:
            output = logsumexp_wirelength_cpp.forward(pos.view(pos.numel()), flat_netpin, netpin_start, net_mask, gamma, num_threads)
        ctx.flat_netpin = flat_netpin
        ctx.netpin_start = netpin_start
        ctx.netpin_values = netpin_values
        ctx.net_mask = net_mask
        ctx.gamma = gamma
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3]
        ctx.exp_nxy_sum = output[4]
        ctx.num_threads = num_threads
        ctx.pos = pos
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            output = logsumexp_wirelength_hip.backward(
                    grad_pos,
                    ctx.pos,
                    ctx.exp_xy, ctx.exp_nxy,
                    ctx.exp_xy_sum, ctx.exp_nxy_sum,
                    ctx.flat_netpin,
                    ctx.netpin_start,
                    ctx.netpin_values,
                    ctx.net_mask,
                    ctx.gamma
                    )
        else:
            output = logsumexp_wirelength_cpp.backward(
                    grad_pos,
                    ctx.pos,
                    ctx.exp_xy, ctx.exp_nxy,
                    ctx.exp_xy_sum, ctx.exp_nxy_sum,
                    ctx.flat_netpin,
                    ctx.netpin_start,
                    ctx.net_mask,
                    ctx.gamma,
                    ctx.num_threads
                    )
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        return output, None, None, None, None, None, None

class LogSumExpWirelengthAtomicFunction(Function):
    """compute weighted average wirelength.
    @param pos pin location (x array, y array), not cell location
    @param pin2net_map pin2net map
    @param net_mask whether to compute wirelength
    @param gamma the smaller, the closer to HPWL
    """
    @staticmethod
    def forward(ctx, pos, pin2net_map, net_mask, gamma):
        if pos.is_cuda:
            output = logsumexp_wirelength_hip_atomic.forward(pos.view(pos.numel()), pin2net_map, net_mask, gamma)
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        ctx.pin2net_map = pin2net_map
        ctx.net_mask = net_mask
        ctx.gamma = gamma
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3];
        ctx.exp_nxy_sum = output[4];
        ctx.pos = pos
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            output = logsumexp_wirelength_hip_atomic.backward(
                    grad_pos,
                    ctx.pos,
                    ctx.exp_xy.view([-1]), ctx.exp_nxy.view([-1]),
                    ctx.exp_xy_sum.view([-1]), ctx.exp_nxy_sum.view([-1]),
                    ctx.pin2net_map,
                    ctx.net_mask,
                    ctx.gamma
                    )
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        return output, None, None, None

class LogSumExpWirelength(nn.Module):
    """ Compute log-sum-exp wirelength.
    CPU only supports net-by-net algorithm.
    GPU supports two algorithms: atomic, sparse.
    Different parameters are required for different algorithms.

    @param flat_netpin flat netpin map, length of #pins
    @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins
    @param pin2net_map pin2net map
    @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore
    @param gamma the smaller, the closer to HPWL
    @param algorithm must be net-by-net | atomic | sparse
    """
    def __init__(self, flat_netpin=None, netpin_start=None, pin2net_map=None, net_mask=None, gamma=None, algorithm='atomic', num_threads=8):
        super(LogSumExpWirelength, self).__init__()
        assert net_mask is not None and gamma is not None, "net_mask, gamma are requried parameters"
        if algorithm == 'net-by-net':
            assert flat_netpin is not None and netpin_start is not None, "flat_netpin, netpin_start are requried parameters for algorithm net-by-net"
        elif algorithm == 'atomic':
            assert pin2net_map is not None, "pin2net_map is required for algorithm atomic"
        elif algorithm == 'sparse':
            assert flat_netpin is not None and netpin_start is not None and pin2net_map is not None, "flat_netpin, netpin_start, pin2net_map are requried parameters for algorithm sparse"
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.netpin_values = None
        self.pin2net_map = pin2net_map
        self.net_mask = net_mask
        self.gamma = gamma
        self.algorithm = algorithm
        self.num_threads = num_threads
    def forward(self, pos):
        if pos.is_cuda:
            if self.algorithm == 'atomic':
                return LogSumExpWirelengthAtomicFunction.apply(pos,
                        self.pin2net_map,
                        self.net_mask,
                        self.gamma
                        )
            elif self.algorithm == 'sparse':
                if self.netpin_values is None:
                    self.netpin_values = torch.ones_like(self.flat_netpin, dtype=pos.dtype)
                return LogSumExpWirelengthFunction.apply(pos,
                        self.flat_netpin,
                        self.netpin_start,
                        self.netpin_values,
                        self.net_mask,
                        self.gamma,
                        self.num_threads
                        )
        else: # only net-by-net for CPU
            return LogSumExpWirelengthFunction.apply(pos,
                    self.flat_netpin,
                    self.netpin_start,
                    None,
                    self.net_mask,
                    self.gamma,
                    self.num_threads
                    )
