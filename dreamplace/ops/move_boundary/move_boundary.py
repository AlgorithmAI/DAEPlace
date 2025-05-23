##
# @file   move_boundary.py
# @author Xu Li
# @date   10 2024
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.move_boundary.move_boundary_cpp as move_boundary_cpp
try:
    import dreamplace.ops.move_boundary.move_boundary_hip as move_boundary_hip
except:
    pass

class MoveBoundaryFunction(Function):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
          xl,
          yl,
          xh,
          yh,
          num_movable_nodes,
          num_filler_nodes,
          num_threads
          ):
        if pos.is_cuda:
            output = move_boundary_hip.forward(
                    pos.view(pos.numel()),
                    node_size_x,
                    node_size_y,
                    xl,
                    yl,
                    xh,
                    yh,
                    num_movable_nodes,
                    num_filler_nodes
                    )
        else:
            output = move_boundary_cpp.forward(
                    pos.view(pos.numel()),
                    node_size_x,
                    node_size_y,
                    xl,
                    yl,
                    xh,
                    yh,
                    num_movable_nodes,
                    num_filler_nodes,
                    num_threads
                    )
        return output

class MoveBoundary(object):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """
    def __init__(self, node_size_x, node_size_y, xl, yl, xh, yh, num_movable_nodes, num_filler_nodes, num_threads=8):
        super(MoveBoundary, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_threads = num_threads

    def forward(self, pos):
        return MoveBoundaryFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                xl=self.xl,
                yl=self.yl,
                xh=self.xh,
                yh=self.yh,
                num_movable_nodes=self.num_movable_nodes,
                num_filler_nodes=self.num_filler_nodes,
                num_threads=self.num_threads
                )
    
    def __call__(self, pos):
        return self.forward(pos)