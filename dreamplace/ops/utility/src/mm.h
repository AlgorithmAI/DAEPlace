/**
 * @file   mm.h
 * @author Xu Li
 * @date   6 2024
 */
#ifndef GPUPLACE_MM_H
#define GPUPLACE_MM_H

#include "hipblas.h"

// template specialization for hipblasSgemm and hipblasDgemm
template <typename T>
hipblasStatus_t mm(
        hipblasHandle_t handle,
        hipblasOperation_t transa, hipblasOperation_t transb,
        int m, int  n, int k,
        const T     *alpha,
        const T     *A, int lda,
        const T     *B, int ldb,
        const T     *beta,
        T          *C, int ldc
        );

template <>
hipblasStatus_t mm<float>(
        hipblasHandle_t handle,
        hipblasOperation_t transa, hipblasOperation_t transb,
        int m, int n, int k,
        const float     *alpha,
        const float     *A, int lda,
        const float     *B, int ldb,
        const float     *beta,
        float       *C, int ldc
        )
{
    return hipblasSgemm(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc
            );
}

template <>
hipblasStatus_t mm<double>(
        hipblasHandle_t handle,
        hipblasOperation_t transa, hipblasOperation_t transb,
        int m, int n, int k,
        const double    *alpha,
        const double    *A, int lda,
        const double    *B, int ldb,
        const double    *beta,
        double      *C, int ldc
        )
{
    return hipblasDgemm(
             handle,
             transa, transb,
             m, n, k,
             alpha,
             A, lda,
             B, ldb,
             beta,
             C, ldc
            );
}

#endif
