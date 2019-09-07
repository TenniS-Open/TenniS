//
// Created by kier on 2019-09-03.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/stack.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_roi_align.h"
#include "dragon/roi_align_op.h"

/**
 * THis cpu implement was fork from Dragon, SeetaTech
 */

namespace ts {
    namespace cpu {

        class ROIAlign : public OperatorOnCPU<base::ROIAlign> {
        public:
            std::shared_ptr<dragon::ROIAlignOp<dragon::CPUContext>> m_dragon;

            void init() override {
                dragon::Workspace ws;
                m_dragon = std::make_shared<dragon::ROIAlignOp<dragon::CPUContext>>(this, &ws);
            }

            int run(Stack &stack) override {
                std::vector<Tensor> inputs;
                for (size_t i = 0; i < stack.size(); ++i) {
                    inputs.emplace_back(stack[i]);
                }
                auto memory_device = this->running_memory_device();
                for (auto &input : inputs) {
                    input = input.view(memory_device);
                }
                m_dragon->bind_inputs(inputs);
                m_dragon->bind_outputs(1);

                m_dragon->RunOnDevice();

                stack.push(Tensor::Pack(m_dragon->outputs()));
                m_dragon->clear_inputs();
                m_dragon->clear_outputs();
                return 1;
            }


            std::vector<Tensor> roi_align(
                    const std::vector<Tensor> &inputs,
                    int pool_h, int pool_w, float spatial_scale, int sampling_ratio) override {
                TS_LOG_ERROR << "What a Terrible Failure!" << eject;
                return {};
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ROIAlign, CPU, name::layer::roi_align())
