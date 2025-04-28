#ifndef GPUPLACE_GEAM_H
#define GPUPLACE_GEAM_H

#include "hipblas.h"

//template specialization for hipblasSgeam and hipblasDgeam
template <typename T>
hipblasStatus_t geam(
        hipblasHandle_t handle,
        hipblasOperation_t transa, hipblasOperation_t transb,
        int m, int n,
        const T     *alpha,
        const T     *A, int lda,
        const T     *beta,
        const T     *B, int ldb,
        T           *C, int ldc
        );

template <>
hipblasStatus_t geam<float>(
        hipblasHandle_t handle,
        hipblasOperation_t transa, hipblasOperation_t transb,
        int m, int n,
        const T     *alpha,
        const T     *A, int lda,
        const T     *beta,
        const T     *B, int ldb,
        T           *C, int ldc
        );
{
    return hipblasSgeam(
        handle,
        transa, transb,
        m, n,
        alpha,
        A, lda,
        beta,
        B, ldb,
        C, ldc
            );
}

template <>
hipblasStatus_t geam<double>(
        hipblasHandle_t handle,
        hipblasOperation_t transa, hipblasOperation_t transb,
        int m, int n,
        const double           *alpha,
        const double           *A, int lda,
        const double           *beta,
        const double           *B, int ldb,
        double           *C, int ldc
        )
{
    return hipblasDgeam(
        handle,
        transa, transb,
        m, n,
        alpha,
        A, lda,
        beta,
        B, ldb,
        C, ldc
            );
}

#endif