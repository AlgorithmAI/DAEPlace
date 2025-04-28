#include <stdio.h>
#include <math.h>
#include <float.h>
#include "hip/hip_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeMoveBoundary(
        T* x_tensor,
        const T* node_size_x_tensor,
        const T xl, const T xh,
        const int num_nodes,
        const int num_movable_nodes,
        const int num_filler_nodes
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x * gridDim.x)
    {
        if (i < num_movable_nodes || i >= num_nodes-num_filler_nodes)
        {
            x_tensor[i] = max(xl, x_tensor[i]);
            x_tensor[i] = min(xh-node_size_x_tensor[i], x_tensor[i]);
        }
    }
}

template <typename T>
int computeMoveBoundaryMapHipLauncher(
        T* x_tensor, T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T xl, const T yl, const T xh, const T yh,
        const int num_nodes,
        const int num_movable_nodes,
        const int num_filler_nodes
        )
{
    int block_count = 128;
    int thread_count = 256;

    hipError_t status;
    hipStream_t stream_x;
    hipStream_t stream_y;
    status = hipStreamCreate(&stream_x);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_x\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamCreate(&stream_y);
    if (status != hipSuccess)
    {
        printf("hipStreamCreate failed for stream_y\n");
        fflush(stdout);
        return 1;
    }

   hipLaunchKernelGGL(( computeMoveBoundary), dim3(block_count), dim3(thread_count), 0, stream_x, 
            x_tensor,
            node_size_x_tensor,
            xl, xh,
            num_nodes,
            num_movable_nodes,
            num_filler_nodes
            );

   hipLaunchKernelGGL(( computeMoveBoundary), dim3(block_count), dim3(thread_count), 0, stream_y, 
            y_tensor,
            node_size_y_tensor,
            yl, yh,
            num_nodes,
            num_movable_nodes,
            num_filler_nodes
            );

    /* destroy stream */
    status = hipStreamDestroy(stream_x);
    stream_x = 0;
    if (status != hipSuccess)
    {
        printf("stream_x destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = hipStreamDestroy(stream_y);
    stream_y = 0;
    if (status != hipSuccess)
    {
        printf("stream_y destroy failed\n");
        fflush(stdout);
        return 1;
    }
    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeMoveBoundaryMapLauncher(\
            T* x_tensor, T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T xl, const T yl, const T xh, const T yh, \
            const int num_nodes, \
            const int num_movable_nodes, \
            const int num_filler_nodes \
            )\
    { \
        return computeMoveBoundaryMapHipLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                xl, yl, xh, yh, \
                num_nodes, \
                num_movable_nodes, \
                num_filler_nodes \
                );\
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE