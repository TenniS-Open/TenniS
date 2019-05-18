//
// Created by kier on 19-5-18.
//

#include "declare_operator.h"

#ifdef TS_USE_CUDA

#include "api/operator_gpu.h"
#include "kernels/gpu/gpu_helper.h"

using namespace ts;

class APIPluginOperatorGPU : public Operator {
public:
    using self = APIPluginOperatorGPU;
    using supper = Operator;

    APIPluginOperatorGPU(const self &) = delete;
    APIPluginOperatorGPU &operator=(const self &) = delete;

    APIPluginOperatorGPU(
            const std::string &device, const std::string &op,
            ts_new_Operator *f_new, ts_free_Operator *f_free,
            ts_OperatorGPU_init *f_init, ts_OperatorGPU_infer *f_infer, ts_OperatorGPU_run *f_run)
            : device(device), op(op)
            , f_new(f_new), f_free(f_free)
            , f_init(f_init), f_infer(f_infer), f_run(f_run) {
        obj = this->f_new();
        if (obj == nullptr) {
            TS_LOG_ERROR << "Call ts_new_Operator failed on " << device << " for " << op << eject;
        }
        set_param_checking_mode(ParamCheckingMode::WEAK);
    }

    ~APIPluginOperatorGPU() final {
        if (obj) f_free(obj);
    }

    void init() override {
        auto stream = gpu::get_cuda_stream_on_context();
        ts_OperatorParams params(this);
        f_init(obj, &params, stream);
    }

    std::shared_ptr<ts_Tensor> CallOperatorRun(Stack &stack) {
        auto stream = gpu::get_cuda_stream_on_context();
        APIPluginStack args(stack);
        std::shared_ptr<ts_Tensor> out(f_run(obj, int32_t(args.args.size()), args.args.data(), stream), ts_free_Tensor);
        if (out == nullptr) {
            TS_LOG_ERROR << "Call ts_Operator_run failed (return null) on " << device << " for " << op << eject;
        }
        return out;
    }

    std::shared_ptr<ts_Tensor> CallOperatorInfer(Stack &stack) {
        auto stream = gpu::get_cuda_stream_on_context();
        APIPluginStack args(stack);
        std::shared_ptr<ts_Tensor> out(f_infer(obj, int32_t(args.args.size()), args.args.data(), stream), ts_free_Tensor);
        if (out == nullptr) {
            TS_LOG_ERROR << "Call ts_Operator_infer failed (return null) on " << device << " for " << op << eject;
        }
        return out;
    }

    int run(Stack &stack) override {
        auto out = CallOperatorRun(stack);
        stack.push(**out);
        return 1;
    }

    int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override {
        if (f_infer == nullptr) {
            auto out = CallOperatorRun(stack);
            TensorPrototype proto(**out);
            output = proto.unpack();
        } else {
            auto proto = CallOperatorInfer(stack);
            auto int_array = tensor::cast(INT32, **proto);
            auto size = int_array.count();
            auto data = int_array.data<int32_t>();
            std::vector<Tensor::Prototype> packed_proto;
            class _ : public std::exception {};
            try {
                if (size <= 0) throw _();
                packed_proto.resize(size);
                int anchor = 1;
                for (int i = 0; i < size; ++i) {
                    if (size - anchor < 2) throw _();
                    auto &this_proto = packed_proto[i];
                    auto this_dtype = DTYPE(data[anchor++]);
                    auto this_dims = data[anchor++];
                    if (size - anchor < this_dims) throw _();
                    std::vector<int32_t> this_shape(this_dims);
                    for (int j = 0; j < this_dims; ++j) {
                        this_shape[j] = data[anchor++];
                    }
                    this_proto = Tensor::Prototype(this_dtype, this_shape);
                }
                output = packed_proto;
            } catch (const _&) {
                TS_LOG_ERROR << "Call ts_Operator_infer failed (format error) on " << device << " for " << op << eject;
            }
        }
        return 1;
    }

private:
    void *obj = nullptr;

    std::string device;
    std::string op;

    ts_new_Operator *f_new = nullptr;
    ts_free_Operator *f_free = nullptr;
    ts_OperatorGPU_init *f_init = nullptr;
    ts_OperatorGPU_infer *f_infer = nullptr;
    ts_OperatorGPU_run *f_run = nullptr;
};

void ts_OperatorGPU_Register(const char *op, ts_new_Operator *f_new, ts_free_Operator *f_free,
                          ts_OperatorGPU_init *f_init, ts_OperatorGPU_infer *f_infer, ts_OperatorGPU_run *f_run) {
    TRY_HEAD
    if (f_new == nullptr || f_free == nullptr || f_init == nullptr || f_run == nullptr) {
        TS_LOG_ERROR << "f_new, f_free, f_init and f_run can't be nullptr" << eject;
    }
    std::string cpp_device(ts::GPU);
    std::string cpp_op(op);
    auto creator = [=]() -> Operator::shared {
        return std::make_shared<APIPluginOperatorGPU>(cpp_device, cpp_op,
                f_new, f_free,
                f_init, f_infer, f_run);
    };
    OperatorCreator::Register(cpp_device, cpp_op, creator);
    TRY_TAIL
}

#else

typedef void ts_OperatorGPU_init(void *op, const ts_OperatorParams *dict, void *stream);
typedef ts_Tensor *ts_OperatorGPU_infer(void *op, int32_t argc, ts_Tensor **argv, void *stream);
typedef ts_Tensor *ts_OperatorGPU_run(void *op, int32_t argc, ts_Tensor **argv, void *stream);

using namespace ts;

void ts_OperatorGPU_Register(const char *device, ts_new_Operator *f_new, ts_free_Operator *f_free,
                          ts_OperatorGPU_init *f_init, ts_OperatorGPU_infer *f_infer, ts_OperatorGPU_run *f_run) {
    TRY_HEAD
    TS_LOG_ERROR << "TensorStack not compiled with TS_USE_CUDA, registration failed." << eject;
    TRY_TAIL
}

#endif
