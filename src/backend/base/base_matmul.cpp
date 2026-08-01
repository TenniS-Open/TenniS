#include "backend/base/base_matmul.h"

#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "module/bubble.h"

#include <numeric>


namespace ts {
    namespace base {

        void MatMul::init() {
            supper::init();

            auto &context = ctx::ref<DeviceContext>();

            m_op_inner_prod = OperatorCreator::Create(context.computing_device.type(), "inner_prod", false);
            TS_CHECK_NQ(m_op_inner_prod, nullptr) << "Can not find operator: " << "inner_prod";
            // bubble param
            m_op_inner_prod->set(Bubble::RetentionParam::op, tensor::from(name::layer::gemm()));
            m_op_inner_prod->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_inner_prod->has(param) && this->has(param)) {
                    m_op_inner_prod->set(param, get(param));
                }
            }
            Tensor transb_tensor = tensor::build(ts::BOOLEAN, false);
            m_op_inner_prod->set(name::transpose, transb_tensor);
            m_op_inner_prod->init();
        }

        inline bool is_batch(const Tensor::Prototype &a, const Tensor::Prototype &b,
                      int *out_batch = nullptr, int *out_m = nullptr, int *out_k = nullptr, int *out_n = nullptr) {
            if (a.dims() != b.dims()) return false;
            if (a.dims() < 2) return false;
            auto m = a.size(a.dims() - 2);
            auto k1 = a.size(a.dims() - 1);
            auto k2 = b.size(b.dims() - 2);
            auto n = b.size(b.dims() - 1);
            if (k1 != k2) return false;
            int batch = 1;
            for (int i = 0; i < a.dims() - 2; ++i) {
                if (a.size(i) != b.size(i)) return false;
                batch *= a.size(i);
            }
            if (batch == 1) return false;
            if (out_batch) *out_batch = batch;
            if (out_m) *out_m = m;
            if (out_k) *out_k = k1;
            if (out_n) *out_n = n;
            return true;
        }

        int MatMul::infer(Stack &stack, std::vector<Tensor::Prototype> &outputs) {
            auto a = stack[0];
            auto b = stack[1];

            {
                int batch, m, k, n;
                if (is_batch(a.proto(), b.proto(), &batch, &m, &k, &n)) {
                    outputs.resize(1);
                    Shape output_shape = a.sizes();
                    output_shape[a.dims() - 1] = n;
                    outputs[0] = Tensor(a.dtype(), output_shape).proto();
                    return 1;
                }
            }

            auto dim4 = false;
            if (b.dims() == 4) {
                const auto one = a.size(0) * a.size(1) * b.size(0) * b.size(1);
                if (one != 1) {
                    TS_LOG_ERROR << "Input of MatMal could be [1, 1, ?, k] and [1, 1, k, ?], but got "
                        << a.dims() << " " << b.dims() << eject;
                }
                dim4 = true;
                a = a.reshape({a.size(2), a.size(3)});
                b = b.reshape({b.size(2), b.size(3)});
            } else if (b.dims() > 2) {
                TS_LOG_ERROR << "Dimension of b in MatMal must less than 2, but got " << b.dims() << eject;
            }

            outputs.resize(1);

            if (a.dims() == b.dims()) {
                outputs[0] = Tensor(a.dtype(), {a.sizes()[0], b.sizes()[1]}).proto();
            } else {
                Shape output_shape = a.sizes();
                output_shape[a.dims() - 1] = b.sizes()[1];
                outputs[0] = Tensor(a.dtype(), output_shape).proto();
            }

            if (dim4) {
                auto p = outputs[0];
                outputs[0] = Tensor(p.dtype(), {1, 1, p.size(0), p.size(1)}).proto();
            }

            return 1;
        }

        void run_batch(MatMul *p, Stack &stack, Tensor a, Tensor b, Tensor &out, int batch, int m, int k, int n) {
            // resize output
            out = Tensor(a.dtype(), { batch, m, n });
            a = a.reshape({batch, m, k});
            b = b.reshape({batch, k, n});
            for (int i = 0; i < batch; ++i) {
                Tensor cc;
                Tensor aa = a.slice(i);
                Tensor bb = b.slice(i);
                p->matmul_compute(stack, aa, bb, cc);
                auto dst = out.slice(i).weak_memory();
                auto src = cc.weak_memory();
                auto size = cc.proto().count() * cc.proto().type_bytes();
                memcpy(dst, src, size);
            }
            Shape output_shape = a.sizes();
            output_shape[a.dims() - 1] = n;
            out = out.reshape(output_shape);
        }

        int MatMul::run(Stack &stack) {
            TS_CHECK_GE(stack.size(), 2);

            std::vector<Tensor::Prototype> outputs;

            infer(stack, outputs);

            auto memory_device = running_memory_device();
            auto a = stack[0].view(memory_device);
            auto b = stack[1].view(memory_device);

            {
                int batch, m, k, n;
                if (is_batch(a.proto(), b.proto(), &batch, &m, &k, &n)) {
                    Tensor out;
                    run_batch(this, stack, a, b, out, batch, m, k, n);
                    stack.push(out);
                    return 1;
                }
            }

            auto dim4 = a.dims() == 4 && b.dims() == 4;
            if (dim4) {
                a = a.reshape({a.size(2), a.size(3)});
                b = b.reshape({b.size(2), b.size(3)});
            }

            Tensor out;
            matmul_compute(stack, a, b, out);

            if (dim4) {
                out = out.reshape({1, 1, out.size(0), out.size(1)});
            }

            stack.push(out);

            return 1;
        }
    }
}
