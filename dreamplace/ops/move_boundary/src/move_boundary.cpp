/**
 * @file   move_boundary.cpp
 * @author Xu Li
 * @date   10 2024
 * @brief  Move out-of-bound cells back to inside placement region
 */

#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeMoveBoundaryMapLauncher(
    T* x_tensor, T* y_tensor,
    const T* node_size_x_tensor, const T* node_size_y_tensor,
    const T xl, const T yl, const T xh, const T yh,
    const int num_nodes,
    const int num_movable_nodes,
    const int num_filler_nodes,
    const int num_threads
);

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor move_boundary_forward(
    at::Tensor pos,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    double xl,
    double yl,
    double xh,
    double yh,
    int num_movable_nodes,
    int num_filler_nodes,
    int num_threads
)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    //
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeMoveBoundaryMapLauncher", [&] {
        computeMoveBoundaryMapLauncher<scalar_t>(
            pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2,
            node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(),
            xl, yl, xh, yh,
            pos.numel()/2,
            num_movable_nodes,
            num_filler_nodes,
            num_threads
        );
    });
    return pos;
}

template <typename T>
int computeMoveBoundaryMapLauncher(
        T* x_tensor, T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T xl, const T yl, const T xh, const T yh,
        const int num_nodes,
        const int num_movable_nodes,
        const int num_filler_nodes,
        const int num_threads
        )
{
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nodes; ++i)
    {
        if (i < num_movable_nodes || i >= num_nodes-num_filler_nodes)
        {
            x_tensor[i] = std::max(x_tensor[i], xl);
            x_tensor[i] = std::min(x_tensor[i], xh-node_size_x_tensor[i]);

            y_tensor[i] = std::max(y_tensor[i], yl);
            y_tensor[i] = std::min(y_tensor[i], yh-node_size_y_tensor[i]);
        }
    }

    return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::move_boundary_forward, "MoveBoundary forward");
  //m.def("backward", &DREAMPLACE_NAMESPACE::move_boundary_backward, "MoveBoundary backward");
}
