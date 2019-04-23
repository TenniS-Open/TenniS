//
// Created by kier on 2019-04-13.
//

#include <core/tensor_builder.h>
#include "frontend/intime.h"
#include "runtime/operator.h"
#include "global/operator_factory.h"

#include "runtime/stack.h"

#include "utils/ctxmgr_lite.h"
#include "backend/name.h"
#include "module/menu.h"


namespace ts {
    namespace intime {
        Tensor run(Workbench &bench, const Bubble &bubble, const std::vector<Tensor> &inputs) {
            bench.online_run(bubble, inputs);
            auto &stack = bench.stack();
            auto fields_count = stack.size();
            if (fields_count == 1) {
                return stack[0];
            }
            std::vector<Tensor> fields(fields_count);
            for (int i = 0; i < fields_count; ++i) {
                fields[i] = stack[i];
            }
            Tensor output;
            output.pack(fields);
            return std::move(output);
        }

        Tensor run(const Bubble &bubble, const std::vector<Tensor> &inputs) {
            auto bench = ctx::get<Workbench>();
            if (bench == nullptr) {
                TS_LOG_ERROR << "Must bind Workbench before run" << eject;
            }
            return run(*bench, bubble, inputs);
        }

        Tensor resize2d(const Tensor &x, const Tensor &size, desc::ResizeType type) {
            return run(desc::resize2d(type), {x, size});
        }

        Tensor resize2d(const Tensor &x, const std::vector<int32_t> &size, desc::ResizeType type) {
            return resize2d(x, tensor::from(size), type);
        }


        static int run_const_node(const Node &node, Node &compute_node) {
            std::vector<Node> inputs = node.inputs();
            std::vector<Node> new_inputs;
            int  input_num = inputs.size();
            int const_inputs = 0;
            for(int i=0; i<input_num; i++) {
                auto &bubble = inputs[i].bubble();
                if(bubble.op() == Bubble::Variable) {
                    TS_LOG_ERROR << "Not support " << Bubble::Variable << " in this version" << eject;
                }else if(bubble.op() == Bubble::Const) {
                    new_inputs.push_back(inputs[i]);
                    const_inputs++;
                }else if(bubble.op() == Bubble::Parameter) {
                    new_inputs.push_back(inputs[i]);
                }else {
                    std::vector<Node> op_inputs = inputs[i].inputs();
                    int op_const_inputs = 0;

                    for(int k=0; k<op_inputs.size(); k++) {
                        Node tmpnode = op_inputs[k];
                        int n = run_const_node(op_inputs[k], tmpnode);
                        if(n > 0) {
                            op_inputs[k] = tmpnode;
                            op_const_inputs++;
                        }
                    }

                    if((op_const_inputs > 0) && (op_const_inputs == op_inputs.size())) {
                        std::vector<Tensor> values;
                        for(int k=0; k<op_const_inputs; k++) {
                            auto &opbubble = op_inputs[k].bubble();
                            TS_AUTO_CHECK(opbubble.op() == Bubble::Const);
                            values.push_back(opbubble.get(name::value));
                        }

                        Tensor result = intime::run(bubble, values);
                        auto data = ts::bubble::data(bubble.name(), result);
                        const_inputs++;
                        new_inputs.push_back(data);
                    }else
                    {
                        new_inputs.push_back(inputs[i]);
                    }
                }
            }

            Node::Link(compute_node, new_inputs);

            if(compute_node.bubble().op() == Bubble::Const) {
                return 1;
            }

            if((const_inputs > 0) && (const_inputs == input_num)) {
                std::vector<Tensor> vecs;
                for(int i=0; i<input_num; i++ ) {
                    auto &bubble = new_inputs[i].bubble();
                    TS_AUTO_CHECK(bubble.op() == Bubble::Const);
                    vecs.push_back(bubble.get(name::value));
                }

                Tensor result = intime::run(node.bubble(), vecs);
                auto data = ts::bubble::data(node.bubble().name(), result);
                compute_node = data;
                return 1;
            }

            return 0;
        }


        void run_const_nodes(const std::vector<Node> &nodes, std::vector<Node> &output_nodes) {
            for(int i=0; i<nodes.size();  i++) {
                Node tmpnode(nodes[i]);
                run_const_node(nodes[i], tmpnode);
                output_nodes.push_back(tmpnode);
            }
        }


    }
}
