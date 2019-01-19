//
// Created by kier on 2019/1/19.
//

#include <backend/zoo/copy.h>

#include "backend/zoo/copy.h"
#include "runtime/stack.h"
#include "backend/name.h"
#include "global/operator_factory.h"

namespace ts {
    namespace zoo {

        void Copy::init() {
            supper::init();
        }

        int Copy::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            output.resize(1, stack.index(0)->proto());
            return 1;
        }

        int Copy::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            stack.push(-1);

            return 1;
        }
    }
}

using namespace ts::zoo;

TS_REGISTER_OPERATOR(Copy, ts::CPU, ts::name::layer::copy());
