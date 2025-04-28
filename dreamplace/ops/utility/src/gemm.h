#ifndef GPUPLACE_GEMM_H
#define GPUPLACE_GEMM_H

#include "hipsparse.h"
// template specialization for hipsparseScsrgemm and hipsparseDcsrgemm
template <typename T>
hipsparseStatus_t csrgemm(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        hipsparseOperation_t transB,
        int m,
        int n,
        int k,
        const hipsparseMatDescr_t descrA,
        const int nnzA,
        const T *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        const hipsparseMatDescr_t descrB,
        const int nnzB,
        const T *csrSortedValB,
        const int *csrSortedRowPtrB,
        const int *csrSortedColIndB,
        const hipsparseMatDescr_t descrC,
        T *csrSortedValC,
        const int *csrSortedRowPtrC,
        int *csrSortedColIndC);

template <>
hipsparseStatus_t csrgemm<float>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        hipsparseOperation_t transB,
        int m,
        int n,
        int k,
        const hipsparseMatDescr_t descrA,
        const int nnzA,
        const float *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        const hipsparseMatDescr_t descrB,
        const int nnzB,
        const float *csrSortedValB,
        const int *csrSortedRowPtrB,
        const int *csrSortedColIndB,
        const hipsparseMatDescr_t descrC,
        float *csrSortedValC,
        const int *csrSortedRowPtrC,
        int *csrSortedColIndC)
{
    return hipsparseScsrgemm(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            descrA,
            nnzA,
            csrSortedValA,
            csrSortedRowPtrA,
            csrSortedColIndA,
            descrB,
            nnzB,
            csrSortedValB,
            csrSortedRowPtrB,
            csrSortedColIndB,
            descrC,
            csrSortedValC,
            csrSortedRowPtrC,
            csrSortedColIndC
            );
}

template <>
hipsparseStatus_t csrgemm<double>(
        hipsparseHandle_t handle,
        hipsparseOperation_t transA,
        hipsparseOperation_t transB,
        int m,
        int n,
        int k,
        const hipsparseMatDescr_t descrA,
        const int nnzA,
        const double *csrSortedValA,
        const int *csrSortedRowPtrA,
        const int *csrSortedColIndA,
        const hipsparseMatDescr_t descrB,
        const int nnzB,
        const double *csrSortedValB,
        const int *csrSortedRowPtrB,
        const int *csrSortedColIndB,
        const hipsparseMatDescr_t descrC,
        double *csrSortedValC,
        const int *csrSortedRowPtrC,
        int *csrSortedColIndC)
{
    return hipsparseDcsrgemm(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            descrA,
            nnzA,
            csrSortedValA,
            csrSortedRowPtrA,
            csrSortedColIndA,
            descrB,
            nnzB,
            csrSortedValB,
            csrSortedRowPtrB,
            csrSortedColIndB,
            descrC,
            csrSortedValC,
            csrSortedRowPtrC,
            csrSortedColIndC
            );
}

#endif


