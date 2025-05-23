#include <stdio.h>
#include <math.h>
#include <float.h>
#include "hip/hip_runtime.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeHPWLMax(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_pins,
        T* partial_hpwl_x_max
        )
{
    //static_assert(sizeof(T) == 0, "computeHPWLMax is not implemented for this type");
}

template <>
__global__ void computeHPWLMax<int>(
    const int* x,
    const int* pin2net_map,
    const unsigned char* net_mask,
    int num_pins,
    int* partial_hpwl_x_max
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (i < num_pins && net_mask[net_id])
        {   
            atomicMax(&partial_hpwl_x_max[net_id], x[i]);
            //__syncthreads();
        }
    }
}

template <>
__global__ void computeHPWLMax<unsigned long long int>(
    const unsigned long long int* x,
    const int* pin2net_map,
    const unsigned char* net_mask,
    int num_pins,
    unsigned long long int* partial_hpwl_x_max
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (i < num_pins && net_mask[net_id])
        {   
            atomicMax(&partial_hpwl_x_max[net_id], x[i]);
            //__syncthreads();
        }
    }
}

template <typename T>
__global__ void computeHPWLMin(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_pins,
        T* partial_hpwl_x_min
        )
{
    //static_assert(sizeof(T) == 0, "computeHPWLMix is not implemented for this type");
}

template <>
__global__ void computeHPWLMin<int>(
        const int* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_pins,
        int* partial_hpwl_x_min
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (i < num_pins && net_mask[net_id])
        {
            
            atomicMin(&partial_hpwl_x_min[net_id], x[i]);
            //__syncthreads();
        }
    }
}

template <>
__global__ void computeHPWLMin<unsigned long long int>(
        const unsigned long long int* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_pins,
        unsigned long long int* partial_hpwl_x_min
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int net_id = pin2net_map[i];
        if (i < num_pins && net_mask[net_id])
        {
            atomicMin(&partial_hpwl_x_min[net_id], x[i]);
            //__syncthreads();
        }
    }
}

template <typename T>
int computeHPWLHipAtomicLauncher(
        const T* x, const T* y,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        T* partial_hpwl_max,
        T* partial_hpwl_min
        )
{
    const int thread_count = 1024;
    const int block_count = 32;

    hipError_t status;
    hipStream_t stream_x_max;
    status = hipStreamCreate(&stream_x_max);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_x_max\n");
        fflush(stdout);
        return 1;
    }
    hipStream_t stream_x_min;
    status = hipStreamCreate(&stream_x_min);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_x_min\n");
        fflush(stdout);
        return 1;
    }
    hipStream_t stream_y_max;
    status = hipStreamCreate(&stream_y_max);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_y_max\n");
        fflush(stdout);
        return 1;
    }
    hipStream_t stream_y_min;
    status = hipStreamCreate(&stream_y_min);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_y_min\n");
        fflush(stdout);
        return 1;
    }

    computeHPWLMax<<<block_count, thread_count, 0, stream_x_max>>>(
            x,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_max
            );

    computeHPWLMin<<<block_count, thread_count, 0, stream_x_min>>>(
            x,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_min
            );

    computeHPWLMax<<<block_count, thread_count, 0, stream_y_max>>>(
            y,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_max+num_nets
            );

    computeHPWLMin<<<block_count, thread_count, 0, stream_y_min>>>(
            y,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_min+num_nets
            );

    /* destroy stream */
    status = hipStreamDestroy(stream_x_max);
    stream_x_max = 0;
    if (status != hipSuccess)
    {
        printf("stream_x_max destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamDestroy(stream_x_min);
    stream_x_min = 0;
    if (status != hipSuccess)
    {
        printf("stream_x_min destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamDestroy(stream_y_max);
    stream_y_max = 0;
    if (status != hipSuccess)
    {
        printf("stream_y_max destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamDestroy(stream_y_min);
    stream_y_min = 0;
    if (status != hipSuccess)
    {
        printf("stream_y_min destroy failed\n");
        fflush(stdout);
        return 1;
    }
    //printArray(partial_hpwl, num_nets, "partial_hpwl");

    // I move out the summation to use ATen
    // significant speedup is observed
    //sumArray<<<1, 1>>>(partial_hpwl, num_nets, hpwl);

    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(type) \
    int instantiateComputeHPWLAtomicLauncher(\
        const type* x, const type* y, \
        const int* pin2net_map, \
        const unsigned char* net_mask, \
        int num_nets, \
        int num_pins, \
        type* partial_hpwl_max, \
        type* partial_hpwl_min \
        ) \
    { \
        return computeHPWLHipAtomicLauncher(x, y, \
                pin2net_map, \
                net_mask, \
                num_nets, \
                num_pins, \
                partial_hpwl_max, \
                partial_hpwl_min \
                ); \
    }

REGISTER_KERNEL_LAUNCHER(int);
REGISTER_KERNEL_LAUNCHER(long long int);

DREAMPLACE_END_NAMESPACE
