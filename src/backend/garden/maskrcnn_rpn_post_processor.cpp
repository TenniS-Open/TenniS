//
// Created by kier on 19-4-18.
//

#include "backend/garden/maskrcnn_rpn_post_processor.h"

namespace ts {
    namespace maskrcnn {
        namespace base {
            std::string RPNPostProcessor::Name() {
                return "maskrcnn:rpn_post_processor";
            }

            void RPNPostProcessor::init() {
                supper::init();
            }

            int RPNPostProcessor::infer(Stack &stack, std::vector<Tensor::Prototype> &outputs) {
                return 0;
            }

            int RPNPostProcessor::run(Stack &stack) {
                return 0;
            }
        }
    }
}
