#include "backend/base/base_lstm.h"
#include "core/tensor_builder.h"
#include "module/bubble.h"


namespace ts {
    namespace base {
        LSTM::LSTM() {
            field("direction", REQUIRED);
            field("hidden_size", REQUIRED);
        }
        void LSTM::init() {
            supper::init();

            if (has("direction")) {
                auto dir_str = tensor::to_string(get("direction"));
                m_direction = dir_str == "bidirectional" ? 2 : 0;
            }
            if (has("hidden_size")) {
                m_hidden_size = tensor::to_int(get("hidden_size"));
            }

            auto &context = ctx::ref<DeviceContext>();

            m_op_gemm_notran = OperatorCreator::Create(context.computing_device.type(), name::layer::gemm(), false);
            TS_CHECK_NQ(m_op_gemm_notran, nullptr) << "Can not find operator: " << name::layer::gemm();

            // bubble param
            m_op_gemm_notran->set(Bubble::RetentionParam::op, tensor::from(name::layer::gemm()));
            m_op_gemm_notran->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_gemm_notran->has(param) && this->has(param)) {
                    m_op_gemm_notran->set(param, get(param));
                }
            }
            m_op_gemm_notran->init();

            m_op_gemm_transb = OperatorCreator::Create(context.computing_device.type(), name::layer::gemm(), false);
            TS_CHECK_NQ(m_op_gemm_transb, nullptr) << "Can not find operator: " << name::layer::gemm();

            // bubble param
            m_op_gemm_transb->set(Bubble::RetentionParam::op, tensor::from(name::layer::gemm()));
            m_op_gemm_transb->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_gemm_transb->has(param) && this->has(param)) {
                    m_op_gemm_transb->set(param, get(param));
                }
            }

            Tensor transb_tensor = tensor::build(ts::BOOLEAN, true);
            m_op_gemm_transb->set(name::transB, transb_tensor);
            m_op_gemm_transb->init();


            m_op_mul = OperatorCreator::Create(context.computing_device.type(), name::layer::mul(), false);
            TS_CHECK_NQ(m_op_mul, nullptr) << "Can not find operator: " << name::layer::mul();

            m_op_mul->set(Bubble::RetentionParam::op, tensor::from(name::layer::mul()));
            m_op_mul->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_mul->has(param) && this->has(param)) {
                    m_op_mul->set(param, get(param));
                }
            }
            m_op_mul->init();


            m_op_add = OperatorCreator::Create(context.computing_device.type(), name::layer::add(), false);
            TS_CHECK_NQ(m_op_add, nullptr) << "Can not find operator: " << "add";

            m_op_add->set(Bubble::RetentionParam::op, tensor::from("add"));
            m_op_add->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_add->has(param) && this->has(param)) {
                    m_op_add->set(param, get(param));
                }
            }
            m_op_add->init();


            m_op_tanh = OperatorCreator::Create(context.computing_device.type(), "tanh", false);
            TS_CHECK_NQ(m_op_tanh, nullptr) << "Can not find operator: " << "tanh";

            m_op_tanh->set(Bubble::RetentionParam::op, tensor::from("tanh"));
            m_op_tanh->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_tanh->has(param) && this->has(param)) {
                    m_op_tanh->set(param, get(param));
                }
            }
            m_op_tanh->init();


            m_op_sigmoid = OperatorCreator::Create(context.computing_device.type(), name::layer::sigmoid(), false);
            TS_CHECK_NQ(m_op_sigmoid, nullptr) << "Can not find operator: " << name::layer::sigmoid();

            m_op_sigmoid->set(Bubble::RetentionParam::op, tensor::from(name::layer::sigmoid()));
            m_op_sigmoid->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_sigmoid->has(param) && this->has(param)) {
                    m_op_sigmoid->set(param, get(param));
                }
            }
            m_op_sigmoid->init();


            m_op_concat_dim0 = OperatorCreator::Create(context.computing_device.type(), name::layer::concat(), false);
            TS_CHECK_NQ(m_op_concat_dim0, nullptr) << "Can not find operator: " << name::layer::concat();

            // bubble param
            m_op_concat_dim0->set(Bubble::RetentionParam::op, tensor::from(name::layer::concat()));
            m_op_concat_dim0->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_concat_dim0->has(param) && this->has(param)) {
                    m_op_concat_dim0->set(param, get(param));
                }
            }

            Tensor concat_dim0 = tensor::build(ts::INT8, 0);
            m_op_concat_dim0->set(name::dim, concat_dim0);
            m_op_concat_dim0->init();


            m_op_concat_dim1 = OperatorCreator::Create(context.computing_device.type(), name::layer::concat(), false);
            TS_CHECK_NQ(m_op_concat_dim1, nullptr) << "Can not find operator: " << name::layer::concat();

            // bubble param
            m_op_concat_dim1->set(Bubble::RetentionParam::op, tensor::from(name::layer::concat()));
            m_op_concat_dim1->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_concat_dim1->has(param) && this->has(param)) {
                    m_op_concat_dim1->set(param, get(param));
                }
            }

            Tensor concat_dim1 = tensor::build(ts::INT8, 1);
            m_op_concat_dim1->set(name::dim, concat_dim1);
            m_op_concat_dim1->init();

        }

        int LSTM::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto x = stack[0];
            auto w = stack[1];
            auto r = stack[2];

            output.resize(3);
            int num_directions = m_direction == 2 ? 2 : 1;
            auto y = Tensor(x.dtype(), {x.size(0), num_directions, x.size(1), m_hidden_size});
            auto y_h = Tensor(x.dtype(), {num_directions, x.size(1), m_hidden_size});
            auto y_c = Tensor(x.dtype(), {num_directions, x.size(1), m_hidden_size});

            output[0] = y.proto();
            output[1] = y_h.proto();
            output[2] = y_c.proto();

            return 1;
        }

        int LSTM::run(Stack &stack) {
            TS_CHECK_GE(stack.size(), 3);

            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto x = stack[0];
            auto w = stack[1];
            auto r = stack[2];
            auto b = stack[3];
            auto initial_h = stack[4];
            auto initial_c = stack[5];

            if (initial_h.sizes().size() < 3) {
                std::vector<float> vf(w.size(0) * x.size(1) * r.size(2), 0);
                initial_h = tensor::build(FLOAT32, vf).reshape({w.size(0), x.size(1), r.size(2)});
            }
            if (initial_c.sizes().size() < 3) {
                std::vector<float> vf(w.size(0) * x.size(1) * r.size(2), 0);
                initial_c = tensor::build(FLOAT32, vf).reshape({w.size(0), x.size(1), r.size(2)});
            }

            auto memory_device = running_memory_device();

            x = x.view(memory_device);
            w = w.view(memory_device);
            r = r.view(memory_device);
            b = b.view(memory_device);
            initial_h = initial_h.view(memory_device);
            initial_c = initial_c.view(memory_device);

            std::vector<Tensor> vTensor_f(3);  // forward [y, y_h, y_c]
            std::vector<Tensor> vTensor_r(3);  // reverse [y, y_h, y_c]

            std::vector<Tensor> vout;
            vout.resize(3);

            if (m_direction == 0) {
                auto w_forward = w.slice(0);
                auto r_forward = r.slice(0);
                auto b_forward = b.slice(0);
                auto init_h_forward = initial_h.slice(0);
                auto init_c_forward = initial_c.slice(0);
                lstm_compute(stack, x, w_forward, r_forward, b_forward, init_h_forward, init_c_forward, 0, vTensor_f);
            } else if (m_direction == 1) {
                auto w_reverse = w.slice(1);
                auto r_reverse = r.slice(1);
                auto b_reverse = b.slice(1);
                auto init_h_reverse = initial_h.slice(1);
                auto init_c_reverse = initial_c.slice(1);
                lstm_compute(stack, x, w_reverse, r_reverse, b_reverse, init_h_reverse, init_c_reverse, 1, vTensor_r);
            } else if (m_direction == 2) {
                auto w_forward = w.slice(0);
                auto r_forward = r.slice(0);
                auto b_forward = b.slice(0);
                auto init_h_forward = initial_h.slice(0);
                auto init_c_forward = initial_c.slice(0);
                auto w_reverse = w.slice(1);
                auto r_reverse = r.slice(1);
                auto b_reverse = b.slice(1);
                auto init_h_reverse = initial_h.slice(1);
                auto init_c_reverse = initial_c.slice(1);
                lstm_compute(stack, x, w_forward, r_forward, b_forward, init_h_forward, init_c_forward, 0, vTensor_f);
                lstm_compute(stack, x, w_reverse, r_reverse, b_reverse, init_h_reverse, init_c_reverse, 1, vTensor_r);
            }
            stack.erase(-1);  // erase latest Ht in Stack

            Tensor expanded_forward;
            Tensor expanded_forward_y_h;
            Tensor expanded_forward_y_c;
            if (m_direction != 1) {
                Shape new_forward_shape = vTensor_f[0].sizes();
                new_forward_shape.insert(new_forward_shape.begin() + 1, 1);
                expanded_forward = vTensor_f[0].reshape(new_forward_shape);

                Shape new_forward_shape_y_h = vTensor_f[1].sizes();
                new_forward_shape_y_h.insert(new_forward_shape_y_h.begin(), 1);
                expanded_forward_y_h = vTensor_f[1].reshape(new_forward_shape_y_h);

                Shape new_forward_shape_y_c = vTensor_f[2].sizes();
                new_forward_shape_y_c.insert(new_forward_shape_y_c.begin(), 1);
                expanded_forward_y_c = vTensor_f[2].reshape(new_forward_shape_y_c);
            }

            Tensor expanded_reverse;
            Tensor expanded_reverse_y_h;
            Tensor expanded_reverse_y_c;
            if (m_direction > 0) {
                Shape new_reverse_shape = vTensor_r[0].sizes();
                new_reverse_shape.insert(new_reverse_shape.begin() + 1, 1);
                expanded_reverse = vTensor_r[0].reshape(new_reverse_shape);

                Shape new_reverse_shape_y_h = vTensor_r[1].sizes();
                new_reverse_shape_y_h.insert(new_reverse_shape_y_h.begin(), 1);
                expanded_reverse_y_h = vTensor_r[1].reshape(new_reverse_shape_y_h);

                Shape new_reverse_shape_y_c = vTensor_r[2].sizes();
                new_reverse_shape_y_c.insert(new_reverse_shape_y_c.begin(), 1);
                expanded_reverse_y_c = vTensor_r[2].reshape(new_reverse_shape_y_c);
            }

            if (m_direction == 2) {
                stack.push(expanded_forward);
                stack.push(expanded_reverse);
                RunOperator(m_op_concat_dim1, stack, 2);
                vout[0] = *stack.top();
                stack.erase(-1);

                stack.push(expanded_forward_y_h);
                stack.push(expanded_reverse_y_h);
                RunOperator(m_op_concat_dim0, stack, 2);
                vout[1] = *stack.top();
                stack.erase(-1);

                stack.push(expanded_forward_y_c);
                stack.push(expanded_reverse_y_c);
                RunOperator(m_op_concat_dim0, stack, 2);
                vout[2] = *stack.top();
                stack.erase(-1);
            } else if (m_direction == 0) {
                vout[0] = expanded_forward;
                vout[1] = expanded_forward_y_h;
                vout[2] = expanded_forward_y_c;
            } else if (m_direction == 1) {
                vout[0] = expanded_reverse;
                vout[1] = expanded_reverse_y_h;
                vout[2] = expanded_reverse_y_c;
            }

            Tensor packed;
            packed.pack(vout);
            stack.push(packed);

            return 1;
        }
    }
}