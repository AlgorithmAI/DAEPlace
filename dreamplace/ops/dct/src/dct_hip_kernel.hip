#include <stdio.h>
#include <math.h>
#include <float.h>
#include "hip/hip_runtime.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeMulExpk(
        const T* x,
        const T* expk,
        const int M,
        const int N,
        T* z
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int fft_onesided_size = (N>>1)+1;
        int fft_onesided_size_2x = fft_onesided_size<<1;

        if (col_2x <= N)
        {
            int j = row*fft_onesided_size_2x + col_2x;
            //printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1, col_2x+1, i);
            z[i] = x[j]*expk[col_2x] + x[j+1]*expk[col_2x+1];
        }
        else
        {
            int j = row*fft_onesided_size_2x + (N<<1) - col_2x;
            //printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1, col_2x+1, i);
            z[i] = x[j]*expk[col_2x] - x[j+1]*expk[col_2x+1];
        }
    }
}

template <typename T>
void computeMulExpkHipLauncher(
        const T* x,
        const T* expk,
        const int M,
        const int N,
        T* z
        )
{
    const int thread_count = 256;
    const int block_count = 128;

    computeMulExpk<<<block_count, thread_count>>>(
            x,
            expk,
            M,
            N,
            z
            );
}

template <typename T>
__global__ void computeReorder(
        const T* x,
        const int M,
        const int N,
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int ii = i%N;

        if (ii < (N>>1))
        {
            // i*2
            //printf("x[%d] = y[%d]\n", i+ii, i);
            y[i] = x[i+ii];
        }
        else
        {
            // (N-i)*2-1
            //printf("x[%d] = y[%d]\n", i+N*2-ii*3-1, i);
            y[i] = x[i+N*2-ii*3-1];
        }
    }
}

template <typename T>
void computeReorderHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        )
{
    const int thread_count = 256;
    const int block_count = 128;

    computeReorder<<<block_count, thread_count>>>(
            x,
            M,
            N,
            y
            );
}

template <typename T>
__global__ void computeVk(
        const T* x,
        const T* expk,
        const int M,
        const int N,
        T* v
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*(N/2+1); i += blockDim.x * gridDim.x)
    {
        int ncol = N/2+1;
        int row = i/ncol; // row
        int col = i-row*ncol; // column
        int col_2x = (col<<1);

        // real
        T real = x[row*N+col];
        T imag = (col == 0)? 0 : -x[row*N+N-col];

        v[2*i] = real*expk[col_2x] - imag*expk[col_2x+1];
        // imag, x[N-i]
        v[2*i+1] = real*expk[col_2x+1] + imag*expk[col_2x];
    }

}

template <typename T>
void computeVkHipLauncher(
        const T* x,
        const T* expk,
        const int M,
        const int N,
        T* v
        )
{
    const int thread_count = 256;
    const int block_count = 128;

    computeVk<<<block_count, thread_count>>>(
            x,
            expk,
            M,
            N,
            v
            );
}


template <typename T>
__global__ void computeReorderReverse(
        const T* y,
        const int M,
        const int N,
        T* z
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int row = i/N; // row
        int col = i-row*N; // column

        //printf("z[%d] = y[%d]\n", i, (col&1)? (i-col*3/2+N-1) : (i-col/2));
        //z[i] = (col&1)? y[(i-col*3/2+N-1)] : y[(i-col/2)];
        // according to the paper, it should be N - (col+1)/2 for col is odd
        // but it seems previous implementation accidentally matches this as well
        z[i] = (col&1)? y[(i-col) + N - (col+1)/2] : y[(i-col/2)];
    }
}

template <typename T>
void computeReorderReverseHipLauncher(
        const T* y,
        const int M,
        const int N,
        T* z
        )
{
    const int thread_count = 256;
    const int block_count = 128;

    computeReorderReverse<<<block_count, thread_count>>>(
            y,
            M,
            N,
            z
            );
}

template <typename T>
__global__ void addX0AndScale(
        const T* x,
        const int M,
        const int N,
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int i0 = int(i/N)*N;
        y[i] = (y[i]+x[i0])*0.5;
    }
}

template <typename T>
void addX0AndScaleHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        )
{
    addX0AndScale<<<128, 256>>>(
            x,
            M,
            N,
            y
            );
}

/// extends from addX0AndScale to merge scaling
template <typename T>
__global__ void addX0AndScaleN(
        const T* x,
        const int M,
        const int N,
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int i0 = int(i/N)*N;
        // this is to match python implementation
        // normal way should be multiply by 0.25*N
        y[i] = y[i]*0.25*N+x[i0]*0.5;
    }
}

template <typename T>
void addX0AndScaleNHipLauncher(
        const T* x,
        const int M,
        const int N,
        T* y
        )
{
    addX0AndScaleN<<<128, 256>>>(
            x,
            M,
            N,
            y
            );
}

template <typename T>
__global__ void computePad(
        const T* x, // M*N
        const int M,
        const int N,
        T* z // M*2N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int j = row*(N<<1) + col;
        z[j] = x[i];
    }
}

template <typename T>
void computePadHipLauncher(
        const T* x, // M*N
        const int M,
        const int N,
        T* z // M*2N
        )
{
    computePad<<<128, 256>>>(
            x,
            M,
            N,
            z
            );
}

template <typename T>
__global__ void computeMulExpk_2N(
        const T* x, // M*(N+1)*2
        const T* expk,
        const int M,
        const int N,
        T* z // M*N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int j = row*((N+1)<<1) + col_2x;
        z[i] = x[j]*expk[col_2x] + x[j+1]*expk[col_2x+1];
    }
}

template <typename T>
void computeMulExpk_2N_HipLauncher(
        const T* x, // M*(N+1)*2
        const T* expk,
        const int M,
        const int N,
        T* z // M*N
        )
{
    computeMulExpk_2N<<<128, 256>>>(
            x,
            expk,
            M,
            N,
            z
            );
}

template <typename T>
__global__ void computeMulExpkAndPad_2N(
        const T* x, // M*N
        const T* expk,
        const int M,
        const int N,
        T* z // M*2N*2
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int j = row*(N<<2) + col_2x;
        z[j] = x[i]*expk[col_2x];
        z[j+1] = x[i]*expk[col_2x+1];
    }
}

template <typename T>
void computeMulExpkAndPad_2N_HipLauncher(
        const T* x, // M*N
        const T* expk,
        const int M,
        const int N,
        T* z // M*2N*2
        )
{
    computeMulExpkAndPad_2N<<<128, 256>>>(
            x,
            expk,
            M,
            N,
            z
            );
}

/// remove last N entries in each column
template <typename T>
__global__ void computeTruncation(
        const T* x, // M*2N
        const int M,
        const int N,
        T* z // M*N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x)
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int j = row*(N<<1) + col;
        z[i] = x[j];
    }
}

template <typename T>
void computeTruncationHipLauncher(
        const T* x, // M*2N
        const int M,
        const int N,
        T* z // M*N
        )
{
    computeTruncation<<<128, 256>>>(
            x,
            M,
            N,
            z
            );
}

// manually instantiate the template function
#define REGISTER_MULPEXPK_KERNEL_LAUNCHER(type) \
    template void computeMulExpkHipLauncher<type>( \
        const type* x, \
        const type* expk, \
        const int M, \
        const int N, \
        type* z \
        );


REGISTER_MULPEXPK_KERNEL_LAUNCHER(float);
REGISTER_MULPEXPK_KERNEL_LAUNCHER(double);

#define REGISTER_REORDER_KERNEL_LAUNCHER(type) \
    template void computeReorderHipLauncher<type>( \
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        );


REGISTER_REORDER_KERNEL_LAUNCHER(float);
REGISTER_REORDER_KERNEL_LAUNCHER(double);

#define REGISTER_VK_KERNEL_LAUNCHER(type) \
    template void computeVkHipLauncher<type>( \
        const type* x, \
        const type* expk, \
        const int M, \
        const int N, \
        type* v \
        );


REGISTER_VK_KERNEL_LAUNCHER(float);
REGISTER_VK_KERNEL_LAUNCHER(double);

#define REGISTER_REORDERREVERSE_KERNEL_LAUNCHER(type) \
    template void computeReorderReverseHipLauncher<type>( \
        const type* y, \
        const int M, \
        const int N, \
        type* z \
        );


REGISTER_REORDERREVERSE_KERNEL_LAUNCHER(float);
REGISTER_REORDERREVERSE_KERNEL_LAUNCHER(double);

#define REGISTER_ADDX0ANDSCALE_KERNEL_LAUNCHER(type) \
    template void addX0AndScaleHipLauncher<type>( \
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        );


REGISTER_ADDX0ANDSCALE_KERNEL_LAUNCHER(float);
REGISTER_ADDX0ANDSCALE_KERNEL_LAUNCHER(double);

#define REGISTER_ADDX0ANDSCALEN_KERNEL_LAUNCHER(type) \
    template void addX0AndScaleNHipLauncher<type>( \
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        );


REGISTER_ADDX0ANDSCALEN_KERNEL_LAUNCHER(float);
REGISTER_ADDX0ANDSCALEN_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTEPAD_KERNEL_LAUNCHER(type) \
    template void computePadHipLauncher<type>( \
        const type* x, \
        const int M, \
        const int N, \
        type* z \
        );


REGISTER_COMPUTEPAD_KERNEL_LAUNCHER(float);
REGISTER_COMPUTEPAD_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTEMULEXPK_2N_KERNEL_LAUNCHER(type) \
    template void computeMulExpk_2N_HipLauncher<type>( \
        const type* x, \
        const type* expk, \
        const int M, \
        const int N, \
        type* z \
        );


REGISTER_COMPUTEMULEXPK_2N_KERNEL_LAUNCHER(float);
REGISTER_COMPUTEMULEXPK_2N_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTEMULEXPKANDPAD_2N_KERNEL_LAUNCHER(type) \
    template void computeMulExpkAndPad_2N_HipLauncher<type>( \
        const type* x, \
        const type* expk, \
        const int M, \
        const int N, \
        type* z \
        );


REGISTER_COMPUTEMULEXPKANDPAD_2N_KERNEL_LAUNCHER(float);
REGISTER_COMPUTEMULEXPKANDPAD_2N_KERNEL_LAUNCHER(double);

#define REGISTER_COMPUTETRUNCATION_KERNEL_LAUNCHER(type) \
    template void computeTruncationHipLauncher<type>( \
        const type* x, \
        const int M, \
        const int N, \
        type* z \
        );



REGISTER_COMPUTETRUNCATION_KERNEL_LAUNCHER(float);
REGISTER_COMPUTETRUNCATION_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
