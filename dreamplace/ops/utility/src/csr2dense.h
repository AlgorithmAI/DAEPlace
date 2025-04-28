#ifndef GPUPLACE_CSR2DENSE_H
#define GPUPLACE_CSR2DENSE_H

#include "hipsparse.h"

//template specialization for hipsparseScsr2dense and hipsparseDcsr2dense
template <typename T>
hipsparseStatus_t csr2dense(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipsparseMatDesrc_t descrA,
    const T *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    T *A,
    int lda
    );

template <>
hipsparseStatus_t csr2dense<float>(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipsparseMatDesrc_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float *A,
    int lda
    )
{
    return hipsparseScsr2dense(
            handle,
            m,
            n,
            descrA,
            csrValA,
            csrColIndA,
            A,
            lda
            );
}

template <>
cusparseStatus_t csr2dense<double>(
        cusparseHandle_t handle,
        int m,
        int n,
        const cusparseMatDescr_t descrA,
        const double *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        double *A,
        int lda
        )
{
    return cusparseDcsr2dense(
            handle,
            m,
            n,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            A,
            lda
            );
}

#endif