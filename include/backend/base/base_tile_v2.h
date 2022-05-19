//
// Created by kier on 2022/5/19.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_TILE_V2_H
#define TENSORSTACK_BACKEND_BASE_BASE_TILE_V2_H

#include "operator_on_device.h"
#include <array>

namespace ts {
    namespace base {
        class TileV2 : public OperatorOnDevice {
        public:
            using self = TileV2;
            using supper = OperatorOnDevice;

            TileV2();

            void init() override;

            /**
             *
             * @param stack x[1...N]
             * @return
             */
            int run(ts::Stack &stack) override;

            int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) override;

            virtual void tile(const Tensor &x, const std::vector<int32_t> &repeats, Tensor &out) = 0;

        private:
            // Shape m_repeats;
            // bool m_zeros;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_TILE_V2_H
