/**
 * @file utils.h
 * @athor Xu Li
 * @data 6 2024
 */
/
#ifndef _DREAMPLAC_UTILITY_UTILS_H
#define _DREAMPLAC_UTILITY_UTILS_H

#include <hip.h>
#include <hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define allocateHIP(var, size, type) \
{\
  hipError_t status = hipMalloc(&(var), (size) * sizeof(type));\
  if(status != hipSuccess)  \
  {\
     printf("hipMalloc failed for var##\n"); \
  }\
}

#define destroyHIP(var)\
{ \
     hipError_t status = hipFree(var);   \
     if(status != hipSuccess)            \
     {                 \
        printf("hipFree faild for var##\n"); \
     }\
}

#define checkHIP(status) \
{                        \
  if(status != hipSuccess){   \
    printf("HIP Runtime Error :%s\n", \
         hipGetErrorString(status));  \
    assert(status == hipSuccess);\
  }                       \
}

#define allocateCopyHIP(var, rhs, size) \
{                                   \
    allocateHIP(var, size, size);       \
    checkHIP(hipMemcpy(var, rhs, sizeof(decltype(*rhs))*(size), hipMemcpyHostToDevice)); \
}

__device__ inline long long int d_get_globaltime(void)
{
    long long int ret;

    asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(ret));

    return ret;
}

//返回当前时间点自纪元，毫秒精度
__device__ inline double d_get_timer_period(void)
{
    return 1.0e-6;
}

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void)
{
    using namespace std::chrono;
    return high_resolution_clock::now().time_since_epoch().count();
}

//返回时间点自纪元，毫秒
inline double get_timer_period(void)
{
    using namespace  std::chrono;
    return 1000.0 * high_resolution_clock::period::num / high_resultion_clock::period::den;
}

#denfine declareHIPKernel(k)    \
    hr_clock_rep k##_time = 0;  \
    int k##_runs = 0;

#define callHIPKernel(k, n_block, n_threads, shared, ...) \
{                                                         \
    timer_start = d_get_globaltimer();                    \
    k <<< n_blocks, n_threads, shared>>>(__VA_ARGS__); \
    checkHIP(hipDeviceSynchronize());\
    timer_stop = d_get_globaltime();\
    k##_timer += timer_stop - timer_start;  \
    k##_run++;                              \
}

#define callHIPKernelAsync(k, n_blocks, n_threads, shared, stream, ...) \
{                                                                       \
    timer_start = d_get_globaltime();					\
	k <<< n_blocks, n_threads,  shared, stream>>> (__VA_ARGS__);			\
	checkHIP(hipDeviceSynchronize());					\
	timer_stop = d_get_globaltime();					\
	k##_time += timer_stop - timer_start;				\
	k##_runs++;                     \
}

#denfine reportHIPKernelStats(k)
    printf(#k "\t %g \t %d \t %g\n", d_get_timer_period() * k##_time, k##_runs, d_get_timer_period() * k##_time / k##_runs);

template <typename T>
inline __device__ T HIPDiv(T a, T b)
{
    return a / b;
}

template <>
inline __device__ float HIPDiv(float a, float b)
{
    return _fdividef(a, b);
}

template <typename T>
inline __device__ T HIPCeilDiv(T a, T b)
{
    return ceil(HIPDiv(a, b));
}

template <>
inline __device__ int HIPCeilDiv(int a, int b)
{
    return HIPDiv(a+b-1, b);
}
template <>
inline __device__ unsigned int HIPCeilDiv(unsigned int a, unsigned int b)
{
    return HIPDiv(a+b-1, b);
}

template <typename T>
inline __host__ T CPUDiv(T a, T b)
{
    return a / b;
}

template <typename T>
inline __host__ T CPUCeilDiv(T a, T b)
{
    return ceil(CPUDiv(a, b));
}

template <>
inline __host__ int CPUCeilDiv(int a, int b)
{
    return CPUDiv(a+b-1, b);
}
template <>
inline __host__ unsigned int CPUCeilDiv(unsigned int a, unsigned int b)
{
    return CPUDiv(a+b-1, b);
}

#endif
