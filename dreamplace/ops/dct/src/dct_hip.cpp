/**
 * @file   dct_hip.cpp
 * @author Xu Li
 * @date   10 2024
 */
#include "dct_hip.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dct_forward(
        at::Tensor x,
        at::Tensor expk)
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk);
    CHECK_CONTIGUOUS(expk);

    auto N = x.size(-1);
    auto M = x.numel()/N;

    //std::cout << "x\n" << x << "\n";
    //auto x_reorder = at::empty_like(x);
    auto x_reorder = at::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct_forward", [&] {
            computeReorderHipLauncher<scalar_t>(
                    x.data<scalar_t>(),
                    M,
                    N,
                    x_reorder.data<scalar_t>()
                    );
            //std::cout << "x_reorder\n" << x_reorder << "\n";

            auto y = at::rfft(x_reorder, 1, false, true);
            y.mul_(1.0/N);
            //std::cout << "y\n" << y << "\n";

            // re-use x_reorder as output
            computeMulExpkHipLauncher<scalar_t>(
                    y.data<scalar_t>(),
                    expk.data<scalar_t>(),
                    M,
                    N,
                    x_reorder.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";
    });

    return x_reorder;
}

at::Tensor idct_forward(
        at::Tensor x,
        at::Tensor expk)
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk);
    CHECK_CONTIGUOUS(expk);

    auto N = x.size(-1);
    auto M = x.numel()/N;

    //std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk is hermitian symmetric, only fill in half
    auto v = at::empty({M*N+std::max(M, N)}, x.options()).resize_({M, N/2+1, 2});

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct_forward", [&] {
            computeVkHipLauncher<scalar_t>(
                    x.data<scalar_t>(),
                    expk.data<scalar_t>(),
                    M,
                    N,
                    v.data<scalar_t>()
                    );

            //std::cout << "x_reorder\n" << x_reorder << "\n";

            // y is real now
            auto y = at::irfft(v, 1, false, true, {N});

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            //auto z = at::empty_like(x);
            //auto z = at::empty({M, N}, x.options());
            // reuse v
            v.resize_({M, N});
            computeReorderReverseHipLauncher(
                    y.data<scalar_t>(),
                    M,
                    N,
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";
            // this is to match python implementation
            // normal way should be multiply by 0.25*N
            v.mul_(0.5*N);
    });

    return v;
}

at::Tensor dct2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1)
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk0);
    CHECK_CONTIGUOUS(expk0);
    CHECK_GPU(expk1);
    CHECK_CONTIGUOUS(expk1);

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    auto N = x.size(-1);
    auto M = x.numel()/N;
    auto x_reorder = at::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_forward", [&] {
            computeReorderHipLauncher<scalar_t>(
                    x.data<scalar_t>(),
                    M,
                    N,
                    x_reorder.data<scalar_t>()
                    );

            //std::cout << "x_reorder\n" << x_reorder << "\n";

            auto y = at::rfft(x_reorder, 1, false, true);
            //y.mul_(1.0/N);
            //std::cout << "y\n" << y << "\n";

            // re-use x_reorder as output
            //std::cout << "expk1\n" << expk1 << "\n";
            computeMulExpkHipLauncher(
                    y.data<scalar_t>(),
                    expk1.data<scalar_t>(),
                    M,
                    N,
                    x_reorder.data<scalar_t>()
                    );
            //std::cout << "z\n" << x_reorder << "\n";

            // 1D DCT to rows
            auto xt = x_reorder.transpose(-2, -1).contiguous();
            //std::cout << "xt\n" << xt << "\n";
            // I do not want to allocate memory another time
            //x_reorder = at::empty_like(xt);
            x_reorder = x_reorder.view_as(xt);
            computeReorderHipLauncher<scalar_t>(
                    xt.data<scalar_t>(),
                    N,
                    M,
                    x_reorder.data<scalar_t>()
                    );

            //std::cout << "x_reorder\n" << x_reorder << "\n";

            y = at::rfft(x_reorder, 1, false, true);
            //y.mul_(1.0/M);
            //std::cout << "y\n" << y << "\n";

            // re-use x_reorder as output
            //std::cout << "expk0\n" << expk0 << "\n";
            computeMulExpkHipLauncher(
                    y.data<scalar_t>(),
                    expk0.data<scalar_t>(),
                    N,
                    M,
                    x_reorder.data<scalar_t>()
                    );

            x_reorder.mul_(1.0/(M*N));
            x_reorder.transpose_(-2, -1);
    });

    return x_reorder.contiguous();
}

at::Tensor idct2_forward(
        at::Tensor x,
        at::Tensor expk0,
        at::Tensor expk1)
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(expk0);
    CHECK_CONTIGUOUS(expk0);
    CHECK_GPU(expk1);
    CHECK_CONTIGUOUS(expk1);

    auto N = x.size(-1);
    auto M = x.numel()/N;

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk is hermitian symmetric, only fill in half
    auto v = at::empty({M*N+std::max(M, N)}, x.options()).resize_({M, N/2+1, 2});

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct2_forward", [&] {
            computeVkHipLauncher<scalar_t>(
                    x.data<scalar_t>(),
                    expk1.data<scalar_t>(),
                    M,
                    N,
                    v.data<scalar_t>()
                    );

            //std::cout << "expk1\n" << expk1 << "\n";

            auto y = at::irfft(v, 1, false, true, {N});
            //y.mul_(0.25*N);

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            //auto z = at::empty_like(x);
            //auto z = at::empty({M, N}, x.options());
            // reuse v
            v.resize_({M, N});
            computeReorderReverseHipLauncher(
                    y.data<scalar_t>(),
                    M,
                    N,
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            // 1D DCT to rows
            auto xt = v.transpose(-2, -1).contiguous();
            //std::cout << "xt\n" << xt << "\n";
            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            // reuse v
            v.resize_({N, M/2+1, 2});
            computeVkHipLauncher<scalar_t>(
                    xt.data<scalar_t>(),
                    expk0.data<scalar_t>(),
                    N,
                    M,
                    v.data<scalar_t>()
                    );

            //std::cout << "expk0\n" << expk0 << "\n";
            //std::cout << "v\n" << v << "\n";

            y = at::irfft(v, 1, false, true, {M});
            //y.mul_(0.25*M);

            //std::cout << "y\n" << y << "\n";

            // I do not want to allocate memory another time
            v.resize_({N, M});
            computeReorderReverseHipLauncher(
                    y.data<scalar_t>(),
                    N,
                    M,
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            // this is to match python implementation
            // normal way should be multiply by 0.25*0.25*M*N
            v.mul_(0.25*M*N);
            v.transpose_(-2, -1);
    });

    return v.contiguous();
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct", &DREAMPLACE_NAMESPACE::dct_forward, "DCT forward (HIP)");
  m.def("idct", &DREAMPLACE_NAMESPACE::idct_forward, "IDCT forward (HIP)");
  m.def("dct2", &DREAMPLACE_NAMESPACE::dct2_forward, "DCT2 forward (HIP)");
  m.def("idct2", &DREAMPLACE_NAMESPACE::idct2_forward, "IDCT2 forward (HIP)");

  m.def("dst", &DREAMPLACE_NAMESPACE::dst_forward, "DST forward (HIP)");
  m.def("idst", &DREAMPLACE_NAMESPACE::idst_forward, "IDST forward (HIP)");

  m.def("idxct", &DREAMPLACE_NAMESPACE::idxct_forward, "IDXCT forward (HIP)");
  m.def("idxst", &DREAMPLACE_NAMESPACE::idxst_forward, "IDXST forward (HIP)");
  m.def("idcct2", &DREAMPLACE_NAMESPACE::idcct2_forward, "IDCCT2 forward (HIP)");
  m.def("idcst2", &DREAMPLACE_NAMESPACE::idcst2_forward, "IDCST2 forward (HIP)");
  m.def("idsct2", &DREAMPLACE_NAMESPACE::idsct2_forward, "IDSCT2 forward (HIP)");

  m.def("dct_2N", &DREAMPLACE_NAMESPACE::dct_2N_forward, "DCT forward (HIP)");
  m.def("idct_2N", &DREAMPLACE_NAMESPACE::idct_2N_forward, "IDCT forward (HIP)");
  m.def("dct2_2N", &DREAMPLACE_NAMESPACE::dct2_2N_forward, "DCT2 forward (HIP)");
  m.def("idct2_2N", &DREAMPLACE_NAMESPACE::idct2_2N_forward, "IDCT2 forward (HIP)");
}

