//
// Created by kier on 2019/4/8.
//

#include <backend/base/base_strided_slice.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        StridedSlice::StridedSlice() {
            field(name::begin, REQUIRED);
            field(name::end, REQUIRED);
            field(name::stride, OPTIONAL);
        }

        void StridedSlice::init() {
            supper::init();

            m_begin = tensor::array::to_int(get(name::begin));
            m_end = tensor::array::to_int(get(name::end));

            if (has(name::stride)) {
                m_stride = tensor::array::to_int(get(name::stride));
            } else {
                m_stride.resize(m_begin.size(), 1);
            }

            TS_AUTO_CHECK(m_begin.size() == m_end.size() && m_end.size() == m_stride.size());
        }

        int infer_output(int x, int &begin, int &end, int &stride) {
            if (begin < 0) begin = x + begin;
            if (end < 0) end = x + end;
            begin = std::max(0, begin);
            end = std::min(x, end);
            TS_AUTO_CHECK(begin < end);
            return (end - begin - 1) / stride + 1;
        }

        static std::vector<int> infer_output(
                const std::vector<int> &x,
                std::vector<int> &begin,
                std::vector<int> &end,
                std::vector<int> &stride) {
            std::vector<int> out(x.size());
            size_t size = begin.size();
            size_t i = 0;
            for (; i < size; ++i) {
                out[i] = infer_output(x[i], begin[i], end[i], stride[i]);
            }
            for (; i < x.size(); ++i) {
                out[i] = x[i];
            }
            return std::move(out);
        }

        int StridedSlice::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto &x = stack[0];
            auto begin = m_begin;
            auto end = m_end;
            auto stride = m_stride;
            auto out = infer_output(x.sizes(), begin, end, stride);
            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), out);

            return 1;
        }

        int StridedSlice::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto begin = m_begin;
            auto end = m_end;
            auto stride = m_stride;
            auto output_shape = infer_output(stack[0].sizes(), begin, end, stride);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto &out = *stack.push(x.dtype(), output_shape, memory_device);

            strided_slice(x, begin, end, stride, out);

            return 1;
        }
    }
}
