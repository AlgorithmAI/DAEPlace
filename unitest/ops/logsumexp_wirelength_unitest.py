##
# @file   logsumexp_wirelength_unitest.py
# @author Xu Li
# @date   10 2024
#

import os
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.logsumexp_wirelength import logsumexp_wirelength
sys.path.pop()

def unsorted_segment_max(pin_x, pin2net_map, num_nets):
    result = torch.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(len(pin2net_map)):
        result[pin2net_map[i]] = result[pin2net_map[i]].max(pin_x[i])
    return result

def unsorted_segment_min(pin_x, pin2net_map, num_nets):
    result = torch.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(len(pin2net_map)):
        result[pin2net_map[i]] = result[pin2net_map[i]].min(pin_x[i])
    return result

def unsorted_segment_sum(pin_x, pin2net_map, num_nets):
    result = torch.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(len(pin2net_map)):
        result[pin2net_map[i]] += pin_x[i]
    return result

def build_wirelength(pin_x, pin_y, pin2net_map, net2pin_map, gamma, ignore_net_degree):
    # wirelength cost 
    # log-sum-exp 
    # ignore_net_degree is not supported yet 

    # temporily store exp(x)
    scaled_pin_x = pin_x/gamma
    scaled_pin_y = pin_y/gamma

    exp_pin_x = torch.exp(scaled_pin_x)
    exp_pin_y = torch.exp(scaled_pin_y)
    nexp_pin_x = torch.exp(-scaled_pin_x)
    nexp_pin_y = torch.exp(-scaled_pin_y)

    # sum of exp(x) 
    sum_exp_pin_x = unsorted_segment_sum(exp_pin_x, pin2net_map, len(net2pin_map))
    sum_exp_pin_y = unsorted_segment_sum(exp_pin_y, pin2net_map, len(net2pin_map))
    sum_nexp_pin_x = unsorted_segment_sum(nexp_pin_x, pin2net_map, len(net2pin_map))
    sum_nexp_pin_y = unsorted_segment_sum(nexp_pin_y, pin2net_map, len(net2pin_map))

    wl = (torch.log(sum_exp_pin_x) + torch.log(sum_nexp_pin_x) + torch.log(sum_exp_pin_y) + torch.log(sum_nexp_pin_y))*gamma

    wirelength = torch.sum(wl)

    return wirelength

class LogSumExpWirelengthOpTest(unittest.TestCase):
    def test_logsumexp_wirelength_random(self):
        pin_pos = np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]], dtype=np.float32)*10
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        pin2net_map = np.zeros(len(pin_pos), dtype=np.int32)
        for net_id, pins in enumerate(net2pin_map):
            for pin in pins:
                pin2net_map[pin] = net_id

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]
        gamma = torch.tensor(0.5, dtype=torch.float32)
        ignore_net_degree = 4

        # net mask 
        net_mask = np.ones(len(net2pin_map), dtype=np.uint8)
        for i in range(len(net2pin_map)):
            if len(net2pin_map[i]) >= ignore_net_degree:
                net_mask[i] = 0 

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count 
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)
        
        print("flat_net2pin_map = ", flat_net2pin_map)
        print("flat_net2pin_start_map = ", flat_net2pin_start_map)

        print(np.transpose(pin_pos))
        pin_pos_var = Variable(torch.from_numpy(np.transpose(pin_pos)).reshape([-1]), requires_grad=True)
        #pin_pos_var = torch.nn.Parameter(torch.from_numpy(np.transpose(pin_pos)).reshape([-1]))
        print(pin_pos_var)

        golden = build_wirelength(pin_pos_var[:pin_pos_var.numel()//2], pin_pos_var[pin_pos_var.numel()//2:], pin2net_map, net2pin_map, gamma, ignore_net_degree)
        print("golden_value = ", golden.data)
        golden.backward()
        golden_grad = pin_pos_var.grad.clone()
        print("golden_grad = ", golden_grad.data)

        # test cpu 
        # clone is very important, because the custom op cannot deep copy the data 
        pin_pos_var.grad.zero_()
        custom = logsumexp_wirelength.LogSumExpWirelength(
                torch.from_numpy(flat_net2pin_map), 
                torch.from_numpy(flat_net2pin_start_map),
                torch.from_numpy(pin2net_map), 
                torch.from_numpy(net_mask), 
                torch.tensor(gamma), 
                algorithm='sparse'
                )
        result = custom.forward(pin_pos_var)
        print("custom = ", result)
        result.backward()
        grad = pin_pos_var.grad.clone()
        print("custom_grad = ", grad)

        np.testing.assert_allclose(result.data.numpy(), golden.data.detach().numpy())
        np.testing.assert_allclose(grad.data.numpy(), golden_grad.data.numpy())

        # test gpu 
        if torch.cuda.device_count(): 
            pin_pos_var.grad.zero_()
            custom_hip = logsumexp_wirelength.LogSumExpWirelength(
                    Variable(torch.from_numpy(flat_net2pin_map)).cuda(), 
                    Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
                    torch.from_numpy(pin2net_map).cuda(), 
                    torch.from_numpy(net_mask).cuda(), 
                    torch.tensor(gamma).cuda(), 
                    algorithm='sparse'
                    )
            result_hip = custom_hip.forward(pin_pos_var.cuda())
            print("custom_hip_result = ", result_hip.data.cpu())
            result_hip.backward()
            grad_hip = pin_pos_var.grad.clone()
            print("custom_grad_hip = ", grad_hip.data.cpu())

            np.testing.assert_allclose(result_hip.data.cpu().numpy(), golden.data.detach().numpy())
            np.testing.assert_allclose(grad_hip.data.cpu().numpy(), grad.data.numpy(), rtol=1e-7, atol=1e-12)

        # test gpu atomic
        if torch.cuda.device_count(): 
            pin_pos_var.grad.zero_()
            custom_hip = logsumexp_wirelength.LogSumExpWirelength(
                    Variable(torch.from_numpy(flat_net2pin_map)).cuda(), 
                    Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
                    torch.from_numpy(pin2net_map).cuda(), 
                    torch.from_numpy(net_mask).cuda(), 
                    torch.tensor(gamma).cuda(), 
                    algorithm='atomic'
                    )
            result_hip = custom_hip.forward(pin_pos_var.cuda())
            print("custom_hip_result atomic = ", result_hip.data.cpu())
            result_hip.backward()
            grad_hip = pin_pos_var.grad.clone()
            print("custom_grad_hip atomic = ", grad_hip.data.cpu())

            np.testing.assert_allclose(result_hip.data.cpu().numpy(), golden.data.detach().numpy())
            np.testing.assert_allclose(grad_hip.data.cpu().numpy(), grad.data.numpy(), rtol=1e-7, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
