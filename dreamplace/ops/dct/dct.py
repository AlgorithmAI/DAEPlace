##
# @file   dct.py
# @author Yibo Lin
# @date   Jun 2018
#

import os
import sys
import numpy as np
import torch
from torch.autograd import Function
from torch import nn

import dreamplace.ops.dct.dct_cpp as dct_cpp
try:
    import dreamplace.ops.dct.dct_hip as dct_hip
except:
    pass

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform

def dct(x, expk, algorithm):
    """compute discrete cosine transformation, DCT II, using N-FFT or 2N-FFT
    yk = \sum_{n=0}^{N-1} x_n cos(pi/N*n*(k+1/2))

    @param x sequence
    @param expk coefficients for post-processing
    @param algorithm algorithm type N | 2N
    """
    if x.is_cuda:
        if algorithm == 'N':
            output = dct_hip.dct(x.view([-1, x.size(-1)]), expk)
        elif algorithm == '2N':
            output = dct_hip.dct_2N(x.view([-1, x.size(-1)]), expk)
    else:
        if algorithm == 'N':
            output = dct_cpp.dct(x.view([-1, x.size(-1)]), expk)
        elif algorithm == '2N':
            output = dct_cpp.dct_2N(x.view([-1, x.size(-1)]), expk)
    return output.view(x.size())

class DCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, algorithm):
        return dct(x, expk, algorithm)

class DCT(nn.Module):
    def __init__(self, expk=None, algorithm='N'):
        super(DCT, self).__init__()
        self.expk = expk
        self.algorithm = algorithm
    def forward(self, x):
        if self.expk is None or self.expk.size(-2) != x.size(-1):
            self.expk = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return DCTFunction.apply(x, self.expk, self.algorithm)

def idct(x, expk, algorithm):
    """Compute inverse discrete cosine transformation, which is also the DCT III, using N-FFT or 2N-FFT
    yk = Re { 1/2*x0 + \sum_{n=1}^{N-1} xn exp(j*pi/(2N)*n*(2k+1)) }
    The actual yk will be scaled by 2 to match other python implementation

    @param x sequence
    @param expk coefficients for pre-processing
    @param algorithm algorithm type N | 2N
    """
    if x.is_cuda:
        if algorithm == 'N':
            output = dct_hip.idct(x.view([-1, x.size(-1)]), expk)
        elif algorithm == '2N':
            output = dct_hip.idct_2N(x.view([-1, x.size(-1)]), expk)
    else:
        if algorithm == 'N':
            output = dct_cpp.idct(x.view([-1, x.size(-1)]), expk)
        elif algorithm == '2N':
            output = dct_cpp.idct_2N(x.view([-1, x.size(-1)]), expk)
    return output.view(x.size())

class IDCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, algorithm):
        return idct(x, expk, algorithm)

class IDCT(nn.Module):
    def __init__(self, expk=None, algorithm='N'):
        super(IDCT, self).__init__()
        self.expk = expk
        self.algorithm = algorithm
    def forward(self, x):
        if self.expk is None or self.expk.size(-2) != x.size(-1):
            self.expk = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDCTFunction.apply(x, self.expk, self.algorithm)

def dct2(x, expk0, expk1, algorithm='N'):
    """compute 2D discrete cosine transformation, using N-FFT or 2N-FFT
    """
    if x.is_cuda:
        if algorithm == 'N':
            output = dct_hip.dct2(x, expk0, expk1)
            #output = dct_hip.dct(dct_cuda.dct(x, expk1).transpose_(dim0=-2, dim1=-1).contiguous(), expk0).transpose_(dim0=-2, dim1=-1).contiguous()
        elif algorithm == '2N':
            output = dct_hip.dct2_2N(x, expk0, expk1)
            #output = dct_cuda.dct_2N(dct_cuda.dct_2N(x, expk1).transpose_(dim0=-2, dim1=-1).contiguous(), expk0).transpose_(dim0=-2, dim1=-1).contiguous()
    else:
        if algorithm == 'N':
            output = dct_cpp.dct2(x, expk0, expk1)
            #output = dct_cpp.dct(dct_cpp.dct(x, expk1).transpose_(dim0=-2, dim1=-1).contiguous(), expk0).transpose_(dim0=-2, dim1=-1).contiguous()
        elif algorithm == '2N':
            output = dct_cpp.dct2_2N(x, expk0, expk1)
    return output

class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, algorithm):
        return dct2(x, expk0, expk1, algorithm)

class DCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None, algorithm='N'):
        super(DCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.algorithm = algorithm
    def forward(self, x):
        if self.expk0 is None or self.expk0.size(-2) != x.size(-2):
            self.expk0 = discrete_spectral_transform.get_expk(x.size(-2), dtype=x.dtype, device=x.device)
        if self.expk1 is None or self.expk1.size(-2) != x.size(-1):
            self.expk1 = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return DCT2Function.apply(x, self.expk0, self.expk1, self.algorithm)

def idct2(x, expk0, expk1, algorithm='N'):
    """compute 2D inverse discrete cosine transformation, using N-FFT or 2N-FFT
    """
    if x.is_cuda:
        if algorithm == 'N':
            output = dct_hip.idct2(x, expk0, expk1)
            #output = dct_cuda.idct(dct_cuda.idct(x, expk1).transpose_(dim0=-2, dim1=-1).contiguous(), expk0).transpose_(dim0=-2, dim1=-1).contiguous()
        elif algorithm == '2N':
            output = dct_hip.idct2_2N(x, expk0, expk1)
    else:
        if algorithm == 'N':
            output = dct_cpp.idct2(x, expk0, expk1)
            #output = dct_cpp.idct(dct_cpp.idct(x, expk1).transpose_(dim0=-2, dim1=-1).contiguous(), expk0).transpose_(dim0=-2, dim1=-1).contiguous()
        elif algorithm == '2N':
            output = dct_cpp.idct2_2N(x, expk0, expk1)
    return output

class IDCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, algorithm):
        return idct2(x, expk0, expk1, algorithm)

class IDCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None, algorithm='N'):
        super(IDCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.algorithm = algorithm
    def forward(self, x):
        if self.expk0 is None or self.expk0.size(-2) != x.size(-2):
            self.expk0 = discrete_spectral_transform.get_expk(x.size(-2), dtype=x.dtype, device=x.device)
        if self.expk1 is None or self.expk1.size(-2) != x.size(-1):
            self.expk1 = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDCT2Function.apply(x, self.expk0, self.expk1, self.algorithm)

def dst(x, expk):
    """compute discrete sine transformation
    yk = \sum_{n=0}^{N-1} x_n cos(pi/N*(n+1/2)*(k+1))
    """
    if x.is_cuda:
        output = dct_hip.dst(x.view([-1, x.size(-1)]), expk)
    else:
        output = dct_cpp.dst(x.view([-1, x.size(-1)]), expk)
    return output.view(x.size())

class DSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk):
        return dst(x, expk)

class DST(nn.Module):
    def __init__(self, expk=None):
        super(DST, self).__init__()
        self.expk = expk
    def forward(self, x):
        if self.expk is None or self.expk.size(-2) != x.size(-1):
            self.expk = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return DSTFunction.apply(x, self.expk)

def idst(x, expk):
    """Compute inverse discrete sine transformation, which is also the DST III
    yk = Im { (-1)^k*x_{N-1}/2 + \sum_{n=0}^{N-2} xn exp(j*pi/(2N)*(n+1)*(2k+1)) }
    The actual yk will be scaled by 2 to match other python implementation
    """
    if x.is_cuda:
        output = dct_hip.idst(x.view([-1, x.size(-1)]), expk)
    else:
        output = dct_cpp.idst(x.view([-1, x.size(-1)]), expk)
    return output.view(x.size())

class IDSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk):
        return idst(x, expk)

class IDST(nn.Module):
    def __init__(self, expk=None):
        super(IDST, self).__init__()
        self.expk = expk
    def forward(self, x):
        if self.expk is None or self.expk.size(-2) != x.size(-1):
            self.expk = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDSTFunction.apply(x, self.expk)

def idxct(x, expk):
    """compute inverse discrete cosine transformation
    This is different from ordinary formulation for IDCT III
    yk = Re { \sum_{n=0}^{N-1} xn exp(j*pi/(2N)*n*(2k+1)) }
    """
    if x.is_cuda:
        output = dct_hip.idxct(x.view([-1, x.size(-1)]), expk)
    else:
        output = dct_cpp.idxct(x.view([-1, x.size(-1)]), expk)
    #output = IDCTFunction.forward(ctx, x, expk)
    #output.add_(x[..., 0].unsqueeze(-1)).mul_(0.5)
    ##output.mul_(0.5).add_(x[..., 0].unsqueeze(-1).mul(0.5))
    return output.view(x.size())

class IDXCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk):
        return idxct(x, expk)

class IDXCT(nn.Module):
    def __init__(self, expk=None):
        super(IDXCT, self).__init__()
        self.expk = expk
    def forward(self, x):
        if self.expk is None or self.expk.size(-2) != x.size(-1):
            self.expk = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDXCTFunction.apply(x, self.expk)

def idxst(x, expk):
    """compute inverse discrete sine transformation
    This is different from ordinary formulation for IDCT III
    yk = Im { \sum_{n=0}^{N-1} xn exp(j*pi/(2N)*n*(2k+1)) }
    """
    if x.is_cuda:
        output = dct_hip.idxst(x.view([-1, x.size(-1)]), expk)
    else:
        output = dct_cpp.idxst(x.view([-1, x.size(-1)]), expk)
    return output.view(x.size())

class IDXSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk):
        return idxst(x, expk)

class IDXST(nn.Module):
    def __init__(self, expk=None):
        super(IDXST, self).__init__()
        self.expk = expk
    def forward(self, x):
        if self.expk is None or self.expk.size(-2) != x.size(-1):
            self.expk = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDXSTFunction.apply(x, self.expk)

def idcct2(x, expk0, expk1):
    """compute inverse discrete cosine-sine transformation
    This is equivalent to idcct(idcct(x)^T)^T
    """
    if x.is_cuda:
        output = dct_hip.idcct2(x.view([-1, x.size(-1)]), expk0, expk1)
    else:
        output = dct_cpp.idcct2(x.view([-1, x.size(-1)]), expk0, expk1)
    return output.view(x.size())

class IDCCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1):
        return idcct2(x, expk0, expk1)

class IDCCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDCCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
    def forward(self, x):
        if self.expk0 is None or self.expk0.size(-2) != x.size(-2):
            self.expk0 = discrete_spectral_transform.get_expk(x.size(-2), dtype=x.dtype, device=x.device)
        if self.expk1 is None or self.expk1.size(-2) != x.size(-1):
            self.expk1 = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDCCT2Function.apply(x, self.expk0, self.expk1)

def idcst2(x, expk0, expk1):
    """compute inverse discrete cosine-sine transformation
    This is equivalent to idxct(idxst(x)^T)^T
    """
    if x.is_cuda:
        output = dct_hip.idcst2(x.view([-1, x.size(-1)]), expk0, expk1)
    else:
        output = dct_cpp.idcst2(x.view([-1, x.size(-1)]), expk0, expk1)
    return output.view(x.size())

class IDCST2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1):
        return idcst2(x, expk0, expk1)

class IDCST2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDCST2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
    def forward(self, x):
        if self.expk0 is None or self.expk0.size(-2) != x.size(-2):
            self.expk0 = discrete_spectral_transform.get_expk(x.size(-2), dtype=x.dtype, device=x.device)
        if self.expk1 is None or self.expk1.size(-2) != x.size(-1):
            self.expk1 = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDCST2Function.apply(x, self.expk0, self.expk1)

def idsct2(x, expk0, expk1):
    """compute inverse discrete cosine-sine transformation
    This is equivalent to idxst(idxct(x)^T)^T
    """
    if x.is_cuda:
        output = dct_hip.idsct2(x.view([-1, x.size(-1)]), expk0, expk1)
    else:
        output = dct_cpp.idsct2(x.view([-1, x.size(-1)]), expk0, expk1)
    return output.view(x.size())

class IDSCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1):
        return idsct2(x, expk0, expk1)

class IDSCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDSCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
    def forward(self, x):
        if self.expk0 is None or self.expk0.size(-2) != x.size(-2):
            self.expk0 = discrete_spectral_transform.get_expk(x.size(-2), dtype=x.dtype, device=x.device)
        if self.expk1 is None or self.expk1.size(-2) != x.size(-1):
            self.expk1 = discrete_spectral_transform.get_expk(x.size(-1), dtype=x.dtype, device=x.device)
        return IDSCT2Function.apply(x, self.expk0, self.expk1)

