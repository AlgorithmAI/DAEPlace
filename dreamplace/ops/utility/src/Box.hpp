#ifndef _DREAMPLAACE_UTLITY_BOX_H
#define _DREAMPLAACE_UTLITY_BOX_H

#include "utility/src/Msg.h"
#include "utility/src/limits.hpp"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Box
{
    T xl;
    T yl;
    T xh;
    T yh;

    __host__ __device__ Box()
    {
        invalidate();
    }

    __host__ __device__ Box(T xxl, T yyl, T xxh, T yyh)
        : xl(xxl)
        , yl(yyl)
        , xh(xxh)
        , yh(yyh)
    {
    }

    ///@brief invalidate the box
    __host__ __device__ void invalidate()
    {
        xl = hip::numeric_limits<T>::max();
        yl = hip::numeric_limits<T>::max();
        xh = hip::numeric_limits<T>::lowest();
        yh = hip::numeric_limits<T>::lowest();
    }

    __host__ __device__ bool valid() const
    {
        return (xl <= xh) && (yl <= yh);
    }


    __host__ __device__ void encompass(T x, T y)
    {
    ///@breif invalidate the box

        xl = min(xl, x);
        xh = max(xh, x);
        yl = min(yl, y);
        yh = min(yh, y);
    }

    __host__ __device__ void encompass(T xxl, T yyl, T xxh, T yyh)
    {


        encompass(xxl, yyl);
        encompass(xxh, yyh);
    }

    __host__ __device__ void bloat(T dx, T dy)
    {
    /// @brief bloat x direction by 2*dx, and y direction by 2*dy
    /// @param dx
    /// @param dy

        xl -= dx;
        xh += dx;
        yl -= dy;
        yh += dy;
    }

     __host__ __device__ bool contains(T x, T y) const
    {
        return xl <= x && x <= xh && yl <= y && y <= yh;
    }

    __host__ __device__ bool contains(T xxl, T yyl, T xxh, T yyh) const
    {
        return contains(xxl, yyl) && contains(xxh, yyh);
    }
    /// @return width of the box
    __host__ __device__ T width() const {return xh-xl;}
    /// @return height of the box
    __host__ __device__ T height() const {return yh-yl;}
    /// @return x coordinate of the center of the box
    __host__ __device__ T center_x() const {return (xl+xh)/2;}
    /// @return y coordinate of the center of the box
    __host__ __device__ T center_y() const {return (yl+yh)/2;}
    /// @return center manhattan distance to another box
    __host__ __device__ T center_distance(const Box& rhs) const
    {
        return fabs(rhs.center_x()-center_x()) + fabs(rhs.center_y()-center_y());
    }
    /// @brief print the box
    __host__ __device__ void print() const
    {
        printf("(%g, %g, %g, %g)\n", (double)xl, (double)yl, (double)xh, (double)yh);
    }


};

DREAMPLACE_END_NAMESPACE

#endif
