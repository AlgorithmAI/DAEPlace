#ifndef GPUPLACE_CSRMM_H
#define GPUPLACE_CSRMM_H

#include "hipsparse.h"

// template specialization for hipsparseScsrmm and hipsparseDcsrmm
template <typename T>
hipsparseStatus_t csrmm(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        int m,
        int n,
        int k,
        int nnz,
        const T *alpha,
        const hipsparseMatDescr_t descrA,
        const T *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const T *B,
        int ldb,
        const T *beta,
        T *C,
        int ldc
        );

template <>
hipsparseStatus_t csrmm<float>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        int m,
        int n,
        int k,
        int nnz,
        const float *alpha,
        const hipsparseMatDescr_t descrA,
        const float *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const float *B,
        int ldb,
        const float *beta,
        float *C,
        int ldc
        )
{
    return hipsparseScsrmm(
            handle,
            transA,
            m,
            n,
            k,
            nnz,
            alpha,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            B,
            ldb,
            beta,
            C,
            ldc
            );
}

template <>
hipsparseStatus_t csrmm<double>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        int m,
        int n,
        int k,
        int nnz,
        const double *alpha,
        const hipsparseMatDescr_t descrA,
        const double *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const double *B,
        int ldb,
        const double *beta,
        double *C,
        int ldc
        )
{
    return hipsparseDcsrmm(
            handle,
            transA,
            m,
            n,
            k,
            nnz,
            alpha,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            B,
            ldb,
            beta,
            C,
            ldc
            );
}

// template specialization for hipsparseScsrmm2 and hipsparseDcsrmm2
template <typename T>
hipsparseStatus_t csrmm2(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        hipsparseOperation_t transB,
        int m,
        int n,
        int k,
        int nnz,
        const T *alpha,
        const hipsparseMatDescr_t descrA,
        const T *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const T *B,
        int ldb,
        const T *beta,
        T *C,
        int ldc
        );

template <>
hipsparseStatus_t csrmm2<float>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        hipsparseOperation_t transB,
        int m,
        int n,
        int k,
        int nnz,
        const float *alpha,
        const hipsparseMatDescr_t descrA,
        const float *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const float *B,
        int ldb,
        const float *beta,
        float *C,
        int ldc
        )
{
    return hipsparseScsrmm2(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            nnz,
            alpha,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            B,
            ldb,
            beta,
            C,
            ldc
            );
}

template <>
hipsparseStatus_t csrmm2<double>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        hipsparseOperation_t transB,
        int m,
        int n,
        int k,
        int nnz,
        const double *alpha,
        const hipsparseMatDescr_t descrA,
        const double *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const double *B,
        int ldb,
        const double *beta,
        double *C,
        int ldc
        )
{
    return hipsparseDcsrmm2(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            nnz,
            alpha,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            B,
            ldb,
            beta,
            C,
            ldc
            );
}

#endif