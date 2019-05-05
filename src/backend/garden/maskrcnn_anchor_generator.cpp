//
// Created by kier on 19-4-18.
//

#include "backend/garden/maskrcnn_anchor_generator.h"

namespace ts {
    namespace maskrcnn {
        namespace base {
            std::string AnchorGenerator::Name() {
                return "maskrcnn:anchor_generator";
            }

            void AnchorGenerator::init() {
                supper::init();
            }

            int AnchorGenerator::infer(Stack &stack, std::vector<Tensor::Prototype> &outputs) {
                return 0;
            }

            int AnchorGenerator::run(Stack &stack) {
                return 0;
            }
        }
    }
}
