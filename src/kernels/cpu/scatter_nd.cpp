#include "kernels/cpu/scatter_nd.h"
#include "backend/name.h"
#include "global/operator_factory.h"


namespace ts {
    namespace cpu {
        void ScatterND::scatter(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) {
            std::memcpy(out.data<float>(), data.data<float>(), data.count() * sizeof(float));

            auto update_indices = indices.sizes();

            for (int i = 0; i < update_indices[0]; ++i) {
                int offset = updates.slice(i).count();
                int out_idx = indices.data<int>(i);
                std::memcpy(out.data<float>() + out_idx * offset,
                            updates.data<float>() + i * offset, offset * sizeof(float));
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ScatterND, ts::CPU, "scatter_nd")
