/**
 * @file   dst_hip.cpp
 * @author Xu Li
 * @date   10 2024
 */
#include "dct_hip.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dst_forward(
        at::Tensor x,
        at::Tensor expk)
{
    auto N = x.size(-1);
    auto M = x.numel()/N;

    //std::cout << "x\n" << x << "\n";
    auto x_reorder = x.clone();

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dst_forward", [&] {
            negateOddEntriesHipLauncher<scalar_t>(
                    x_reorder.data<scalar_t>(),
                    M,
                    N
                    );

            auto y = dct_forward(x_reorder, expk);
            //std::cout << "y\n" << y << "\n";

            computeFlipHipLauncher<scalar_t>(
                    y.data<scalar_t>(),
                    M,
                    N,
                    x_reorder.data<scalar_t>()
                    );
            //std::cout << "z\n" << y << "\n";
            });

    return x_reorder;
}

at::Tensor idst_forward(
        at::Tensor x,
        at::Tensor expk)
{
    auto N = x.size(-1);
    auto M = x.numel()/N;

    //std::cout << "x\n" << x << "\n";
    auto x_reorder = at::empty_like(x);
    auto y = at::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idst_forward", [&] {
            computeFlipHipLauncher<scalar_t>(
                    x.data<scalar_t>(),
                    M,
                    N,
                    x_reorder.data<scalar_t>()
                    );

            y = idct_forward(x_reorder, expk);
            //std::cout << "y\n" << y << "\n";

            negateOddEntriesHipLauncher<scalar_t>(
                    y.data<scalar_t>(),
                    M,
                    N
                    );
            //std::cout << "z\n" << y << "\n";
            });

    return y;
}

DREAMPLACE_END_NAMESPACE
