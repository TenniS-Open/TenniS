#include "kernels/cpu/lstm.h"
#include "core/tensor_builder.h"
#include "frontend/intime.h"

namespace ts {
    namespace cpu {
        void LSTM::lstm_compute(Stack &stack, Tensor &x, Tensor &w, Tensor &r, Tensor &b, Tensor &initial_h, Tensor &initial_c,
                                int reverse, std::vector<Tensor> &out) {
            // slice weights
            auto w_i = w.slice(0, m_hidden_size);
            auto w_o = w.slice(m_hidden_size, m_hidden_size * 2);
            auto w_f = w.slice(m_hidden_size * 2, m_hidden_size * 3);
            auto w_c = w.slice(m_hidden_size * 3, m_hidden_size * 4);

            auto r_i = r.slice(0, m_hidden_size);
            auto r_o = r.slice(m_hidden_size, m_hidden_size * 2);
            auto r_f = r.slice(m_hidden_size * 2, m_hidden_size * 3);
            auto r_c = r.slice(m_hidden_size * 3, m_hidden_size * 4);

            auto wb_i = b.slice(0, m_hidden_size);
            auto wb_o = b.slice(m_hidden_size, m_hidden_size * 2);
            auto wb_f = b.slice(m_hidden_size * 2, m_hidden_size * 3);
            auto wb_c = b.slice(m_hidden_size * 3, m_hidden_size * 4);

            auto rb_i = b.slice(m_hidden_size * 4, m_hidden_size * 5);
            auto rb_o = b.slice(m_hidden_size * 5, m_hidden_size * 6);
            auto rb_f = b.slice(m_hidden_size * 6, m_hidden_size * 7);
            auto rb_c = b.slice(m_hidden_size * 7, m_hidden_size * 8);

            int seq_len = x.size(0);

            auto Ht = initial_h;
            auto Ct = initial_c;

            Tensor output;

            // unroll
            for (int i = 0; i < seq_len; ++i) {
                auto Ht_prev = Ht;
                auto Ct_prev = Ct;

                auto idx = reverse == 0 ? i : seq_len - i - 1;

                stack.push(x.slice(idx));
                stack.push(w_f);
                stack.push(wb_f);
                RunOperator(m_op_gemm_transb, stack, 3);

                stack.push(Ht_prev);
                stack.push(r_f);
                stack.push(rb_f);
                RunOperator(m_op_gemm_transb, stack, 3);

                RunOperator(m_op_add, stack, 2);
                RunOperator(m_op_sigmoid, stack, 1);
                auto ft = stack.top();


                stack.push(x.slice(idx));
                stack.push(w_o);
                stack.push(wb_o);
                RunOperator(m_op_gemm_transb, stack, 3);

                stack.push(Ht_prev);
                stack.push(r_o);
                stack.push(rb_o);
                RunOperator(m_op_gemm_transb, stack, 3);

                RunOperator(m_op_add, stack, 2);
                RunOperator(m_op_sigmoid, stack, 1);
                auto ot = stack.top();


                stack.push(x.slice(idx));
                stack.push(w_i);
                stack.push(wb_i);
                RunOperator(m_op_gemm_transb, stack, 3);

                stack.push(Ht_prev);
                stack.push(r_i);
                stack.push(rb_i);
                RunOperator(m_op_gemm_transb, stack, 3);

                RunOperator(m_op_add, stack, 2);
                RunOperator(m_op_sigmoid, stack, 1);
//                auto it = stack.top();


                stack.push(x.slice(idx));
                stack.push(w_c);
                stack.push(wb_c);
                RunOperator(m_op_gemm_transb, stack, 3);

                stack.push(Ht_prev);
                stack.push(r_c);
                stack.push(rb_c);
                RunOperator(m_op_gemm_transb, stack, 3);

                RunOperator(m_op_add, stack, 2);
                RunOperator(m_op_tanh, stack, 1);
//                auto ct = stack.top();

                RunOperator(m_op_mul, stack, 2);  // Ct = ft * Ct_prev + ((it * ct))
                stack.push(*ft); // ft
                stack.push(Ct_prev);
                RunOperator(m_op_mul, stack, 2);
                RunOperator(m_op_add, stack, 2);  // Ct
                Ct = *stack.top();


                // Ht = ot * tanh(Ct)
                RunOperator(m_op_tanh, stack, 1);
                stack.push(*ot);  // ot
                RunOperator(m_op_mul, stack, 2);
                Ht = *stack.top();
//                std::cout << "addr Ht: " << &Ht << std::endl;


                Shape Ht_shape = Ht.sizes();
                Ht_shape.insert(Ht_shape.begin(), 1);
                Tensor expand_Ht = Ht.reshape(Ht_shape);

                Shape Ct_shape = Ct.sizes();
                Ct_shape.insert(Ct_shape.begin(), 1);
                Tensor expand_Ct = Ct.reshape(Ct_shape);


                if (i == 0) {
                    output = expand_Ht;
                } else {
                    if (reverse == 0) {
                        stack.push(output);
                        stack.push(expand_Ht);
                        RunOperator(m_op_concat_dim0, stack, 2);
                        output = *stack.top();
                    } else {
                        stack.push(expand_Ht);
                        stack.push(output);
                        RunOperator(m_op_concat_dim0, stack, 2);
                        output = *stack.top();
                    }
                }

                stack.erase(7, -1);
            }

            out[0] = output;
            out[1] = Ht;
            out[2] = Ct;
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(LSTM, ts::CPU, "LSTM")
