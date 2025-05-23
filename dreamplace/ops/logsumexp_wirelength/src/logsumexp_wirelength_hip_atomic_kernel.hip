#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "hip/hip_runtime.h"
#include "utility/src/csrmv.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMax(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        V* x_max
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMax(&x_max[net_id], (V)(x[i]));
            __syncthreads();
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMin(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        V* x_min
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMin(&x_min[net_id], (V)(x[i]));
            __syncthreads();
        }
    }
}

template <typename T, typename V>
__global__ void computeExp(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        V* x_max,
        T* exp_x
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_x[i] = exp((x[i]-x_max[net_id])/(*gamma));
        }
    }
}

template <typename T, typename V>
__global__ void computeNegExp(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        V* x_min,
        T* exp_nx
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_nx[i] = exp(-(x[i]-x_min[net_id])/(*gamma));
        }
    }
}

template <typename T>
__global__ void computeExpSum(
        const T* exp_x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        T* exp_x_sum
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicAdd(&exp_x_sum[net_id], exp_x[i]);
            __syncthreads();
        }
    }
}

template <typename T, typename V>
__global__ void computeLogSumExp(
        const T* exp_x_sum,
        const V* x_max,
        const int* pin2net_map,
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
            partial_wl[i] = (*gamma)*log(exp_x_sum[i]) + (T)x_max[i];
        }
    }
}

template <typename T, typename V>
__global__ void computeLogSumNegExp(
        const T* exp_nx_sum,
        const V* x_min,
        const int* pin2net_map,
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
            partial_wl[i] = (*gamma)*log(exp_nx_sum[i]) - (T)x_min[i];
        }
    }
}

template <typename T>
__global__ void computeLogSumExpWirelengthGrad(
        const T* exp_x, const T* exp_nx,
        const T* exp_x_sum, const T* exp_nx_sum,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        const T* grad_tensor,
        T* grad_x_tensor
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            grad_x_tensor[i] = (exp_x[i]/exp_x_sum[net_id] - exp_nx[i]/exp_nx_sum[net_id])*(*grad_tensor);
        }
    }
}

template <typename T, typename V>
int computeLogSumExpWirelengthHipAtomicLauncher(
        const T* x, const T* y,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        T* exp_xy, T* exp_nxy,
        T* exp_xy_sum, T* exp_nxy_sum,
        V* xy_max, V* xy_min,
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
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                grad_tensor,
                grad_x_tensor
                );
        computeLogSumExpWirelengthGrad<<<block_count, thread_count, 0, stream_y_exp>>>(
                exp_xy+num_pins, exp_nxy+num_pins,
                exp_xy_sum+num_nets, exp_nxy_sum+num_nets,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
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

        // compute max/min
        computeMax<<<block_count, thread_count, 0, stream_x_exp>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_max
                );
        computeMin<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_min
                );
        computeMax<<<block_count, thread_count, 0, stream_y_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_max+num_nets
                );
        computeMin<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_min+num_nets
                );

        // compute exp and negative exp
        computeExp<<<block_count, thread_count, 0, stream_x_exp>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_max,
                exp_xy
                );
        computeNegExp<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_min,
                exp_nxy
                );
        computeExp<<<block_count, thread_count, 0, stream_y_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_max+num_nets,
                exp_xy+num_pins
                );
        computeNegExp<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_min+num_nets,
                exp_nxy+num_pins
                );

        // compute exp sum
        computeExpSum<<<block_count, thread_count, 0, stream_x_exp>>>(
                exp_xy,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_xy_sum
                );
        computeExpSum<<<block_count, thread_count, 0, stream_nx_exp>>>(
                exp_nxy,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_nxy_sum
                );
        computeExpSum<<<block_count, thread_count, 0, stream_y_exp>>>(
                exp_xy+num_pins,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_xy_sum+num_nets
                );
        computeExpSum<<<block_count, thread_count, 0, stream_ny_exp>>>(
                exp_nxy+num_pins,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_nxy_sum+num_nets
                );

        // compute log sum exp
        computeLogSumExp<<<block_count, thread_count, 0, stream_x_exp>>>(
                exp_xy_sum,
                xy_max,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl
                );
        computeLogSumNegExp<<<block_count, thread_count, 0, stream_nx_exp>>>(
                exp_nxy_sum,
                xy_min,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl+num_nets
                );

        computeLogSumExp<<<block_count, thread_count, 0, stream_y_exp>>>(
                exp_xy_sum+num_nets,
                xy_max+num_nets,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl+2*num_nets
                );
        computeLogSumNegExp<<<block_count, thread_count, 0, stream_ny_exp>>>(
                exp_nxy_sum+num_nets,
                xy_min+num_nets,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl+3*num_nets
                );

        // I move out the summation to use ATen
        // significant speedup is observed
        //sumArray<<<1, 1>>>(partial_wl, 2*num_nets, wl);

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


#define REGISTER_KERNEL_LAUNCHER(T, V) \
    template int computeLogSumExpWirelengthHipAtomicLauncher<T, V>(\
            const T* x, const T* y, \
            const int* pin2net_map, \
            const unsigned char* net_mask, \
            int num_nets, \
            int num_pins, \
            const T* gamma, \
            T* exp_xy, T* exp_nxy, \
            T* exp_xy_sum, T* exp_nxy_sum,\
            V* xy_max, V* xy_min, \
            T* partial_wl, \
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor \
            );
   
REGISTER_KERNEL_LAUNCHER(float, int);
REGISTER_KERNEL_LAUNCHER(double, int);

DREAMPLACE_END_NAMESPACE
