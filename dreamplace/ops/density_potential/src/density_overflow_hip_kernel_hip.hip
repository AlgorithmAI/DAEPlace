/**
 * @file   density_overflow_hip_kernel.cc
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density overflow according to NTUPlace3 (https://doi.org/10.1109/TCAD.2008.923063).
 *          This is for fixed cells.
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "hip/hip_runtime.h"
#include "utility/src/gemm.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeDensityMap(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        T* density_map_tensor)
{
    // rank-one update density map
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes*num_impacted_bins_x*num_impacted_bins_y; i += blockDim.x * gridDim.x)
    {
        // density overflow function
        auto computeDensityOverflowFunc = [](T x, T node_size, T bin_center, T bin_size){
            return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
        };
        int node_id = i/(num_impacted_bins_x*num_impacted_bins_y);
        int residual_index = i-node_id*num_impacted_bins_x*num_impacted_bins_y;
        // x direction
        int bin_index_xl = int((x_tensor[node_id]-xl)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl+int(residual_index / num_impacted_bins_y);
        if (k+1 > num_bins_x)
        {
            continue;
        }
        // y direction
        int bin_index_yl = int((y_tensor[node_id]-yl)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl+(residual_index % num_impacted_bins_y);
        if (h+1 > num_bins_y)
        {
            continue;
        }

        T px = computeDensityOverflowFunc(x_tensor[node_id], node_size_x_tensor[node_id], bin_center_x_tensor[k], bin_size_x);
        T py = computeDensityOverflowFunc(y_tensor[node_id], node_size_y_tensor[node_id], bin_center_y_tensor[h], bin_size_y);
        // still area
        atomicAdd(&density_map_tensor[k*num_bins_y+h], px*py);
        __syncthreads();
    }
}

template <typename T>
int computeDensityOverflowMapHipLauncher(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        T* density_map_tensor
        )
{
    int block_count = 128;
    int thread_count = 256;

   hipLaunchKernelGGL(( computeDensityMap), dim3(block_count), dim3(thread_count), 0, 0, 
            x_tensor, y_tensor,
            node_size_x_tensor, node_size_y_tensor,
            bin_center_x_tensor, bin_center_y_tensor,
            num_nodes,
            num_bins_x, num_bins_y,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_impacted_bins_x, num_impacted_bins_y,
            density_map_tensor);

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeDensityOverflowMapLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const int num_impacted_bins_x, const int num_impacted_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            T* density_map_tensor\
            )\
    { \
        return computeDensityOverflowMapHipLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                num_impacted_bins_x, num_impacted_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                density_map_tensor\
                );\
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

template <typename T>
__global__ void computeGaussianFilterWeights(
        const int num_bins_x, const int num_bins_y,
        const T sigma,
        T* gaussian_filter_tensor
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_bins_x*num_bins_y; i += blockDim.x * gridDim.x)
    {
        int x = i/num_bins_y;
        int y = i-x*num_bins_y;
        T sigma_square = sigma*sigma;
        T x2_y2 = (x-num_bins_x/2)*(x-num_bins_x) + (y-num_bins_y/2)*(y-num_bins_y);
        //gaussian_filter_tensor[i] = 1.0/(2*M_PI*sigma_square) * exp(-x2_y2/(2*sigma_square));
        gaussian_filter_tensor[i] = 2 * exp(-x2_y2/(2*sigma_square));
    }
}

template <typename T>
int computeGaussianFilterLauncher(
        const int num_bins_x, const int num_bins_y,
        const T sigma,
        T* gaussian_filter_tensor
        )
{
    int block_count = 128;
    int thread_count = 256;

   hipLaunchKernelGGL(( computeGaussianFilterWeights), dim3(block_count), dim3(thread_count), 0, 0, 
            num_bins_x, num_bins_y,
            sigma,
            gaussian_filter_tensor
            );

    return 0;
}

#define REGISTER_GAUSSIAN_FILTER_LAUNCHER(T) \
    int instantiateComputeGaussianFilterLauncher(\
            const int num_bins_x, const int num_bins_y, \
            const T sigma, \
            T* gaussian_filter_tensor\
            )\
    { \
        return computeGaussianFilterLauncher(\
                num_bins_x, num_bins_y, \
                sigma, \
                gaussian_filter_tensor\
                );\
    }
REGISTER_GAUSSIAN_FILTER_LAUNCHER(float);
REGISTER_GAUSSIAN_FILTER_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
