#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "hip/hip_runtime.h"
#include "utility/src/csrmv.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void fillArray(T* x, const int n, const T v)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = v;
    }
}

template <typename T>
__global__ void computeExp(const T* x, const T* nx, const int n, const T* gamma, T* exp_x)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        exp_x[i] = exp(x[i]/(*gamma));
    }
}

template <typename T>
__global__ void computeNegExp(const T* x, const T* nx, const int n, const T* gamma, T* exp_nx)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        exp_nx[i] = exp(-x[i]/(*gamma));
    }
}

template <typename T>
__global__ void computeMaxAndExp(
        const T* x,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        T* x_max,
        T* exp_x
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x)
    {
        x_max[i] = -FLT_MAX;
        if (net_mask[i])
        {
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int jj = flat_netpin[j];
                T xx = x[jj];
                x_max[i] = max(x_max[i], xx);
            }
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int jj = flat_netpin[j];
                T xx = x[jj];
                exp_x[jj] = exp((xx-x_max[i])/(*gamma));
            }
        }
    }
}

template <typename T>
__global__ void computeMinAndNegExp(
        const T* x,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        T* x_min,
        T* exp_nx
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x)
    {
        x_min[i] = FLT_MAX;
        if (net_mask[i])
        {
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int jj = flat_netpin[j];
                T xx = x[jj];
                x_min[i] = min(x_min[i], xx);
            }
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int jj = flat_netpin[j];
                T xx = x[jj];
                exp_nx[jj] = exp(-(xx-x_min[i])/(*gamma));
            }
        }
    }
}

template <typename T>
__global__ void computeLogSumExp(
        const T* exp_x_sum,
        const T* x_max,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        T* partial_wl
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x)
    {
        if (net_mask[i])
        {
            partial_wl[i] = (*gamma)*log(exp_x_sum[i]) + x_max[i];
        }
        else
        {
            partial_wl[i] = 0;
        }
    }
}

template <typename T>
__global__ void computeLogSumNegExp(
        const T* exp_nx_sum,
        const T* x_min,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        T* partial_wl
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x)
    {
        if (net_mask[i])
        {
            partial_wl[i] = (*gamma)*log(exp_nx_sum[i]) - x_min[i];
        }
        else
        {
            partial_wl[i] = 0;
        }
    }
}

template <typename T>
__global__ void sumArray(const T* x, const int n, T* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
    {
        *output = 0;
        for (int j = 0; j < n; ++j)
        {
            *output += x[j];
        }
    }
}

template <typename T>
__global__ void computeLogSumExpWirelengthGrad(
        const T* exp_x, const T* exp_nx,
        const T* exp_x_sum, const T* exp_nx_sum,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        const T* grad_tensor,
        T* grad_x_tensor
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x)
    {
        if (net_mask[i])
        {
            T reciprocal_exp_x_sum = 1.0/exp_x_sum[i];
            T reciprocal_exp_nx_sum = 1.0/exp_nx_sum[i];
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int jj = flat_netpin[j];
                grad_x_tensor[jj] = (exp_x[jj]*reciprocal_exp_x_sum - exp_nx[jj]*reciprocal_exp_nx_sum)*(*grad_tensor);
                //grad_x_tensor[jj] = (exp_x[jj]/exp_x_sum[i] - exp_nx[jj]/exp_nx_sum[i])*(*grad_tensor);
            }
        }
    }
}

template <typename T>
int computeLogSumExpWirelengthHipLauncher(
        const T* x, const T* y,
        const int* flat_netpin,
        const int* netpin_start,
        const T* netpin_values,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        T* exp_xy, T* exp_nxy,
        T* exp_xy_sum, T* exp_nxy_sum,
        T* partial_wl, // wirelength of each net
        const T* grad_tensor,
        T* grad_x_tensor, T* grad_y_tensor // the gradient is partial total wirelength to partial pin position
        )
{
    int thread_count = 256;
    int block_count = 128; // separate x and y

    hipError_t status;
    hipStream_t stream_x_exp;
    hipStream_t stream_nx_exp;
    hipStream_t stream_y_exp;
    hipStream_t stream_ny_exp;
    status = hipStreamCreate(&stream_x_exp);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_x_exp\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamCreate(&stream_y_exp);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_y_exp\n");
        fflush(stdout);
        return 1;
    }

    if (grad_tensor)
    {
        computeLogSumExpWirelengthGrad<<<block_count, thread_count, 0, stream_x_exp>>>(
                exp_xy, exp_nxy,
                exp_xy_sum, exp_nxy_sum,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                grad_tensor,
                grad_x_tensor
                );
        computeLogSumExpWirelengthGrad<<<block_count, thread_count, 0, stream_y_exp>>>(
                exp_xy+num_pins, exp_nxy+num_pins,
                exp_xy_sum+num_nets, exp_nxy_sum+num_nets,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                grad_tensor,
                grad_y_tensor
                );
    }
    else
    {
        status = hipStreamCreate(&stream_nx_exp);
        if (status != hipSuccess)
        {
            printf("hipStreamCreate failed for stream_nx_exp\n");
            fflush(stdout);
            return 1;
        }
        status = hipStreamCreate(&stream_ny_exp);
        if (status != hipSuccess)
        {
            printf("hipStreamCreate failed for stream_ny_exp\n");
            fflush(stdout);
            return 1;
        }

        T* xy_max = nullptr;
        status = hipMalloc((void**)&xy_max, 2*num_nets*sizeof(T));
        if (status != hipSuccess)
        {
            printf("hipMalloc failed for xy_max\n");
            fflush(stdout);
            return 1;
        }
        T* xy_min = nullptr;
        status = hipMalloc((void**)&xy_min, 2*num_nets*sizeof(T));
        if (status != hipSuccess)
        {
            printf("hipMalloc failed for xy_min\n");
            fflush(stdout);
            return 1;
        }

        //T* partial_wl = nullptr;
        //status = cudaMalloc((void**)&partial_wl, 2*num_nets*sizeof(T));
        //if (status != cudaSuccess)
        //{
        //    printf("cudaMalloc failed for partial_wl\n");
        //    fflush(stdout);
        //    return 1;
        //}
        //// be careful, partial_wl is not initialized yet

        T alpha = 1.0;
        T beta = 0.0;

        computeMaxAndExp<<<block_count, thread_count, 0, stream_x_exp>>>(
                x,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                xy_max,
                exp_xy
                );
        computeMinAndNegExp<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                xy_min,
                exp_nxy
                );
        computeMaxAndExp<<<block_count, thread_count, 0, stream_y_exp>>>(
                y,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                xy_max+num_nets,
                exp_xy+num_pins
                );
        computeMinAndNegExp<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                xy_min+num_nets,
                exp_nxy+num_pins
                );

        hipsparseStatus_t sparse_status;
        hipsparseHandle_t handle_x_exp = 0;
        hipsparseHandle_t handle_nx_exp = 0;
        hipsparseHandle_t handle_y_exp = 0;
        hipsparseHandle_t handle_ny_exp = 0;
        hipsparseMatDescr_t descr = 0;
        /* initialize cusparse library */
        sparse_status= hipsparseCreate(&handle_x_exp);
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("HIPSPARSE Library initialization failed\n");
            fflush(stdout);
            return 1;
        }
        sparse_status= hipsparseCreate(&handle_nx_exp);
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("HIPSPARSE Library initialization failed\n");
            fflush(stdout);
            return 1;
        }
        sparse_status= hipsparseCreate(&handle_y_exp);
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("HIPSPARSE Library initialization failed\n");
            fflush(stdout);
            return 1;
        }
        sparse_status= hipsparseCreate(&handle_ny_exp);
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("HIPSPARSE Library initialization failed\n");
            fflush(stdout);
            return 1;
        }
        /* create and setup matrix descriptor */
        sparse_status= hipsparseCreateMatDescr(&descr);
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("Matrix descriptor initialization failed\n");
            fflush(stdout);
            return 1;
        }
        hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL);
        hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO);
        hipsparseSetStream(handle_x_exp, stream_x_exp);
        hipsparseSetStream(handle_nx_exp, stream_nx_exp);
        hipsparseSetStream(handle_y_exp, stream_y_exp);
        hipsparseSetStream(handle_ny_exp, stream_ny_exp);

        csrmv(
                handle_x_exp,
                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                num_nets,
                num_pins,
                num_pins,
                &alpha,
                descr,
                netpin_values,
                netpin_start, flat_netpin,
                exp_xy,
                &beta,
                exp_xy_sum
                );
        csrmv(
                handle_y_exp,
                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                num_nets,
                num_pins,
                num_pins,
                &alpha,
                descr,
                netpin_values,
                netpin_start, flat_netpin,
                exp_xy+num_pins,
                &beta,
                exp_xy_sum+num_nets
                );
        csrmv(
                handle_nx_exp,
                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                num_nets,
                num_pins,
                num_pins,
                &alpha,
                descr,
                netpin_values,
                netpin_start, flat_netpin,
                exp_nxy,
                &beta,
                exp_nxy_sum
                );
        csrmv(
                handle_ny_exp,
                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                num_nets,
                num_pins,
                num_pins,
                &alpha,
                descr,
                netpin_values,
                netpin_start, flat_netpin,
                exp_nxy+num_pins,
                &beta,
                exp_nxy_sum+num_nets
                );

        computeLogSumExp<<<block_count, thread_count, 0, stream_x_exp>>>(
                exp_xy_sum,
                xy_max,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                partial_wl
                );
        computeLogSumNegExp<<<block_count, thread_count, 0, stream_nx_exp>>>(
                exp_nxy_sum,
                xy_min,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                partial_wl+num_nets
                );

        computeLogSumExp<<<block_count, thread_count, 0, stream_y_exp>>>(
                exp_xy_sum+num_nets,
                xy_max+num_nets,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                partial_wl+2*num_nets
                );
        computeLogSumNegExp<<<block_count, thread_count, 0, stream_ny_exp>>>(
                exp_nxy_sum+num_nets,
                xy_min+num_nets,
                flat_netpin,
                netpin_start,
                net_mask,
                num_nets,
                gamma,
                partial_wl+3*num_nets
                );

        /* destroy matrix descriptor */
        sparse_status = hipsparseDestroyMatDescr(descr);
        descr = 0;
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("Matrix descriptor destruction failed\n");
            fflush(stdout);
            return 1;
        }

        /* destroy handle */
        sparse_status = hipsparseDestroy(handle_x_exp);
        handle_x_exp = 0;
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("HIPSPARSE Library release of resources failed\n");
            fflush(stdout);
            return 1;
        }
        sparse_status = hipsparseDestroy(handle_nx_exp);
        handle_nx_exp = 0;
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("HIPSPARSE Library release of resources failed\n");
            fflush(stdout);
            return 1;
        }
        sparse_status = hipsparseDestroy(handle_y_exp);
        handle_y_exp = 0;
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("HIPSPARSE Library release of resources failed\n");
            fflush(stdout);
            return 1;
        }
        sparse_status = hipsparseDestroy(handle_ny_exp);
        handle_ny_exp = 0;
        if (sparse_status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("HIPSPARSE Library release of resources failed\n");
            fflush(stdout);
            return 1;
        }

        // I move out the summation to use ATen
        // significant speedup is observed
        //sumArray<<<1, 1>>>(partial_wl, 2*num_nets, wl);

        if (xy_max)
        {
            cudaFree(xy_max);
            xy_max = nullptr;
        }
        if (xy_min)
        {
            cudaFree(xy_min);
            xy_min = nullptr;
        }
        //if (partial_wl)
        //{
        //    cudaFree(partial_wl);
        //    partial_wl = nullptr;
        //}
        fflush(stdout);

        status = hipStreamDestroy(stream_nx_exp);
        stream_nx_exp = 0;
        if (status != hipSuccess)
        {
            printf("stream_nx_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }
        status = hipStreamDestroy(stream_ny_exp);
        stream_ny_exp = 0;
        if (status != hipSuccess)
        {
            printf("stream_ny_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }
    }

    /* destroy stream */
    status = hipStreamDestroy(stream_x_exp);
    stream_x_exp = 0;
    if (status != hipSuccess)
    {
        printf("stream_x_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamDestroy(stream_y_exp);
    stream_y_exp = 0;
    if (status != hipSuccess)
    {
        printf("stream_y_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }

    return 0;
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computeLogSumExpWirelengthHipLauncher<T>(\
            const T* x, const T* y, \
            const int* flat_netpin, \
            const int* netpin_start, \
            const T* netpin_values, \
            const unsigned char* net_mask, \
            int num_nets,\
            int num_pins,\
            const T* gamma, \
            T* exp_xy, T* exp_nxy, \
            T* exp_xy_sum, T* exp_nxy_sum, \
            T* partial_wl, \
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor  \
            );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
