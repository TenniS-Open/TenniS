//
// Created by kier on 2019-04-13.
//

#ifndef TENSORSTACK_FRONTEND_INTIME_H
#define TENSORSTACK_FRONTEND_INTIME_H

#include "core/tensor.h"
#include "module/bubble.h"
#include "compressor.h"

#include <string>
#include <vector>
#include <module/graph.h>
#include <runtime/stack.h>
#include <runtime/workbench.h>


namespace ts {
    namespace intime {
        /**
         * @note: the returned tensor is using borrowed memory from bench, use clone or free, before bench finalization
         */
        TS_DEBUG_API Tensor run(Workbench &bench, const Bubble &bubble, const std::vector<Tensor> &inputs);

        /**
         * @note: the returned tensor is using borrowed memory from bench, use clone or free, before bench finalization
         */
        TS_DEBUG_API Tensor run(const Bubble &bubble, const std::vector<Tensor> &inputs);

        TS_DEBUG_API Tensor resize(const Tensor &x, const Tensor &size, desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor resize(const Tensor &x, const std::vector<int32_t> &size, desc::ResizeType type = desc::ResizeType::LINEAR);
    }
}

#endif //TENSORSTACK_FRONTEND_INTIME_H
