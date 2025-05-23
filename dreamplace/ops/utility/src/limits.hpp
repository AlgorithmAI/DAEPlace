/**
 * @file  limits.hpp -->limits.h
 * @author Xu Li
 * #@date  6 2024
 */

#ifndef _DREAMPLACE_UTILITY_LIMITS_H
#define _DREAMPLACE_UTILITY_LIMITS_H

#include <limits.h>
#include <float.h>

namespace hip
{
    template <typename T>
    struct numeric_limits_base
    {
        typedef T type;
    };
    template <typename T>
    struct numeric_limits : public numeric_limits_base<T>
    {
    };

    template <>
    struct numeric_limits<char> : public numeric_limits_base<char>
    {
        __host__ __device__ static constexpr type
            min() noexcept { return CHAR_MIN; }

        __host__ __device__ static constexpr type
            max() noexcept { return CHAR_MAX; }

        __host__ __device__ static constexpr type
            lowest() noexcept { return CHAR_MIN}
    };

    template <>
    struct numeric_limits<unsigned char> : public numeric_limits_base<unsigned char>
    {
        __host__ __device__ static constexpr type
            min() noexcept { return 0; }

        __host__ __device__ static constexpr type
            max() noexcept { return 0; }

        __host__ __device__ static  constexpr type
            lowest()  noexcept { return 0; }

    };

    template <>
    struct numeric_limits<short> : public numeric_limits_base<short>
    {
        __host__ __device__ static constexpr type
            min() noexcept { return SHRT_MIN; }

        __host__ __device__ static constexpr type
            max() noexcept { return SHRT_MAX; }

        __host__ __device__ static constexpr type
            lowest() noexcept { return SHRT_MIN; }
    };

    template <>
    struct numeric_limits<unsigned short> : numeric_limits_base<unsigned short>
    {
        __host__ __device__ static constexpr type
            min() noexcept { return 0; }

        __host__ __device__ static constexpr type
            max() noexcept { return USHRT_MAX; }

        __host__ __device__ static constexpr type
            lowest() noexcept { return 0; }
    };

    template <>
    struct numeric_limits<int> : public numeric_limits_base<int>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return INT_MIN; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return INT_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return INT_MIN; }
    };

    template <>
    struct numeric_limits<unsigned int> : public numeric_limits_base<unsigned int>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return 0; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return UINT_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return 0; }
    };

    template <>
    struct numeric_limits<long> : public numeric_limits_base<long>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return LONG_MIN; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return LONG_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return LONG_MIN; }
    };

    template <>
    struct numeric_limits<unsigned long> : public numeric_limits_base<unsigned long>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return 0; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return ULONG_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return 0; }
    };

    template <>
    struct numeric_limits<long long> : public numeric_limits_base<long long>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return LLONG_MIN; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return LLONG_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return LLONG_MIN; }
    };

    template <>
    struct numeric_limits<unsigned long long> : public numeric_limits_base<unsigned long long>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return 0; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return ULLONG_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return 0; }
    };

    template <>
    struct numeric_limits<float> : public numeric_limits_base<float>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return FLT_MIN; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return FLT_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return -FLT_MAX; }
    };

    template <>
    struct numeric_limits<double> : public numeric_limits_base<double>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return DBL_MIN; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return DBL_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return -DBL_MAX; }
    };

    template <>
    struct numeric_limits<long double> : public numeric_limits_base<long double>
    {
        /** The minimum finite value, or for floating types with
          denormalization, the minimum positive normalized value.  */
        __host__ __device__ static constexpr type
            min() noexcept { return LDBL_MIN; }

        /** The maximum finite value.  */
        __host__ __device__ static constexpr type
            max() noexcept { return LDBL_MAX; }

        /** A finite value x such that there is no other finite value y
         *  where y < x.  */
        __host__ __device__ static constexpr type
            lowest() noexcept { return -LDBL_MAX; }
    };
}

#endif