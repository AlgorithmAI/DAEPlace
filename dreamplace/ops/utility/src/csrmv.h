#ifndef GPUPLACE_CSRMV_H
#define GPUPLACE_CSRMV_H

#include "hipsparse.h"

// template specialization for hipsparseScsrmv and hipsparseDcsrmv
template <typename T>
hipsparseStatus_t csrmv(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        int m,
        int n,
        int nnz,
        const T *alpha,
        const hipsparseMatDescr_t descrA,
        const T *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        const T *x,
        const T *beta,
        T *y
);

template <>
hipsparseStatus_t csrmv<float>(
        hipsparseHandle_t  handle,
        hipsparseOperation_t transA,
        int m,
        int n,
        int nnz,
        const float *alpha,
        const hipsparseMatDescr_t descrA,
        const float *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        const float *x,
        const float *beta,
        float *y
)
{
    return hipsparseScsrmv(
            handle,
            transA,
            m,
            n,
            nnz,
            alpha,
            descrA,
            csrSortedValA,
            csrSortedRowPtrA,
            csrSortedColIndA,
            x,
            beta,
            y
            );
}

template <>
hipsparseStatus_t csrmv<double>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        int m,
        int n,
        int nnz,
        const double *alpha,
        const hipsparseMatDescr_t descrA,
        const double *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        const double *x,
        const double *beta,
        double *y
)
{
    return hipsparseDcsrmv(
            handle,
            transA,
            m,
            n,
            nnz,
            alpha,
            descrA,
            csrSortedValA,
            csrSortedRowPtrA,
            csrSortedColIndA,
            x,
            beta,
            y
            );
}

#endif