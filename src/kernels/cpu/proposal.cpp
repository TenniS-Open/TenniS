//
// Created by kier on 2019-09-03.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/stack.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_proposal.h"

#include "dragon/proposal_op.h"

/**
 * THis cpu implement was fork from Dragon, SeetaTech
 */

namespace ts {
    namespace cpu {
        class Proposal : public OperatorOnCPU<base::Proposal> {
        public:
            std::shared_ptr<dragon::ProposalOp<dragon::CPUContext>> m_dragon;
            int m_output_size;

            void init() override {
                dragon::Workspace ws;
                m_dragon = std::make_shared<dragon::ProposalOp<dragon::CPUContext>>(this, &ws);
                auto max_level = tensor::to_int(get("max_level"));
                auto min_level = tensor::to_int(get("min_level"));
                m_output_size = max_level - min_level + 1;
            }

            int run(Stack &stack) override {
                std::vector<Tensor> inputs;
                for (size_t i = 0; i < stack.size(); ++i) {
                    inputs.emplace_back(stack[i]);
                }
                m_dragon->bind_inputs(inputs);
                m_dragon->bind_outputs(m_output_size);

                m_dragon->RunOnDevice();

                stack.push(Tensor::Pack(m_dragon->outputs()));

                m_dragon->clear_inputs();
                m_dragon->clear_outputs();
                return 1;
            }


            std::vector<Tensor> proposal(
                    const std::vector<Tensor> &inputs,
                    const std::vector<int32_t> &strides,
                    const std::vector<float> &ratios,
                    const std::vector<float> &scales,
                    int32_t pre_nms_top_n,
                    int32_t post_nms_top_n,
                    float nms_thresh,
                    int32_t min_size,
                    int32_t min_level,
                    int32_t max_level,
                    int32_t canonical_scale,
                    int32_t canonical_level) override {
                TS_LOG_ERROR << "What a Terrible Failure!" << eject;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Proposal, CPU, name::layer::proposal())
