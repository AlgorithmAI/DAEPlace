/**
 * @file   dct_cuda.h
 * @author Yibo Lin
 * @date   Sep 2018
 */
#ifndef DREAMPLACE_DCT_HIP_H
#define DREAMPLACE_DCT_HIP_H

#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_GPU(x) AT_ASSERTM(x.is_cuda(), #x "must be a tensor on GPU")
#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor dct_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor idct_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor dct2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

at::Tensor idct2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

at::Tensor dst_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor idst_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor idxct_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor idxst_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor idcct2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

at::Tensor idcst2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

at::Tensor idsct2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

template <typename T>
void computeReorderHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        );

template <typename T>
void computeMulExpkHipLauncher(
        const T* x,
        const T* expk,
        const int M,
        const int N,
        T* z
        );

template <typename T>
void computeVkHipLauncher(
        const T* x,
        const T* expk,
        const int M,
        const int N,
        T* v
        );

template <typename T>
void computeReorderReverseHipLauncher(
        const T* y,
        const int M,
        const int N,
        T* z
        );

template <typename T>
void addX0AndScaleHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        );

/// extends from addX0AndScale to merge scaling
template <typename T>
void addX0AndScaleNHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        );

/// given an array
/// x_0, x_1, ..., x_{N-1}
/// convert to
/// x_{N-1}, ..., x_2, x_1, x_0
template <typename T>
void computeFlipHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        );

/// given an array
/// x_0, x_1, ..., x_{N-1}
/// convert to
/// 0, x_{N-1}, ..., x_2, x_1
/// drop x_0
template <typename T>
void computeFlipAndShiftHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        );

/// flip sign of odd entries
/// index starts from 0
template <typename T>
void negateOddEntriesHipLauncher(
        T* x,
        const int M,
        const int N
        );

at::Tensor dct_2N_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor idct_2N_forward(
        at::Tensor x,
        at::Tensor expk);

at::Tensor dct2_2N_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

at::Tensor idct2_2N_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1);

template <typename T>
void computePadHipLauncher(
        const T* x, // M*N
        const int M,
        const int N,
        T* z // M*2N
        );

template <typename T>
void computeMulExpk_2N_HipLauncher(
        const T* x, // M*(N+1)*2
        const T* expk,
        const int M,
        const int N,
        T* z // M*N
        );

template <typename T>
void computeMulExpkAndPad_2N_HipLauncher(
        const T* x, // M*N
        const T* expk,
        const int M,
        const int N,
        T* z // M*2N*2
        );

/// remove last N entries in each column
template <typename T>
void computeTruncationHipLauncher(
        const T* x, // M*2N
        const int M,
        const int N,
        T* z // M*N
        );

DREAMPLACE_END_NAMESPACE

#endif
