// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "gemm_kernel_base.h"
#include <vector>

namespace kernel_selector {
class GemmKernelTiled : public GemmKernelBase {
public:
    using Parent = GemmKernelBase;

    mutable struct GemmTuningData {
        size_t simd_size = 8;
        size_t tile_m_size = 1;
        size_t tile_k_size = 1;
        size_t tile_n_size = 1;
    } tuning_data;

    GemmKernelTiled() : GemmKernelBase("gemm_nn_tiled") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }
    bool Validate(const Params& params, const optional_params& options) const override;
    DispatchData SetDefault(const gemm_params& params) const override;
    JitConstants GetJitConstants(const gemm_params& params) const override;
    // GemmTuningData InitGemmTuningData(const gemm_params& params) const;
    GemmTuningData SetTuningParams(const gemm_params& params) const;
};
}  // namespace kernel_selector
