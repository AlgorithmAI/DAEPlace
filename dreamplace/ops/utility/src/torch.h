#ifndef _DREAMPLACE_UTILITY_TORCH_H
#define _DREAMPLACE_UTILITY_TORCH_H

#if TORCH_MAJOR_VERSION >= 1
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif
#include <limits>
#include "utility/src/torch_fft_api.h"

#endif