#include "kernels/cpu/scatter_nd.h"
#include "backend/name.h"
#include "global/operator_factory.h"


namespace ts {
    namespace cpu {
        void ScatterND::scatter(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) {
            auto bytes = type_bytes(data.dtype());

            std::memcpy(out.data(), data.data(), data.count() * bytes);

            auto update_indices = indices.sizes();

            for (int i = 0; i < update_indices[0]; ++i) {
                int offset = updates.slice(i).count();
                auto block = offset * bytes;
                int out_idx = indices.data<int>(i);
                std::memcpy(out.data<uint8_t>() + out_idx * block,
                            updates.data<uint8_t>() + i * block, block);
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ScatterND, ts::CPU, "scatter_nd")
