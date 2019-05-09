//
// Created by kier on 19-5-8.
//

#include <core/tensor_builder.h>
#include "declaration.h"

#include "api/operator.h"
#include "global/operator_factory.h"

#include "declare_tensor.h"

#include "runtime/stack.h"

using namespace ts;

struct ts_OperatorParams {
public:
    ts_OperatorParams(Operator *op) : op(op) {}

    Operator *op;
};

class APIPluginStack {
public:
    APIPluginStack(Stack &stack) {
        int argc = stack.size();
        try {
            for (int i = 0; i < argc; ++i) {
                auto &tensor = stack[i];
                args.emplace_back(new ts_Tensor(tensor));
            }
        } catch (const Exception &e) {
            for (auto &arg : args) delete arg;
            throw e;
        } catch (const std::exception &e) {
            for (auto &arg : args) delete arg;
            throw e;
        }
    }

    ~APIPluginStack() {
        for (auto &arg : args) {
            delete arg;
        }
    }

    std::vector<ts_Tensor *> args;
};

class APIPluginOperator : public Operator {
public:
    using self = APIPluginOperator;
    using supper = Operator;

    APIPluginOperator(const self &) = delete;
    APIPluginOperator &operator=(const self &) = delete;

    APIPluginOperator(
            const std::string &device, const std::string &op,
            ts_new_Operator *f_new, ts_free_Operator *f_free,
            ts_Operator_init *f_init, ts_Operator_infer *f_infer, ts_Operator_run *f_run)
            : device(device), op(op)
            , f_new(f_new), f_free(f_free)
            , f_init(f_init), f_infer(f_infer), f_run(f_run) {
        obj = this->f_new();
        if (obj == nullptr) {
            TS_LOG_ERROR << "Call ts_new_Operator failed on " << device << " for " << op << eject;
        }
    }

    ~APIPluginOperator() final {
        if (obj) f_free(obj);
    }

    void init() override {
        ts_OperatorParams params(this);
        f_init(obj, &params);
    }

    std::shared_ptr<ts_Tensor> CallOperatorRun(Stack &stack) {
        APIPluginStack args(stack);
        std::shared_ptr<ts_Tensor> out(f_run(obj, int32_t(args.args.size()), args.args.data()), ts_free_Tensor);
        if (out == nullptr) {
            TS_LOG_ERROR << "Call ts_Operator_run failed (return null) on " << device << " for " << op << eject;
        }
        return out;
    }

    std::shared_ptr<ts_Tensor> CallOperatorInfer(Stack &stack) {
        APIPluginStack args(stack);
        std::shared_ptr<ts_Tensor> out(f_infer(obj, int32_t(args.args.size()), args.args.data()), ts_free_Tensor);
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
    ts_Operator_init *f_init = nullptr;
    ts_Operator_infer *f_infer = nullptr;
    ts_Operator_run *f_run = nullptr;
};

void ts_Operator_Register(const char *device, const char *op, ts_new_Operator *f_new, ts_free_Operator *f_free,
                          ts_Operator_init *f_init, ts_Operator_infer *f_infer, ts_Operator_run *f_run) {
    TRY_HEAD
    if (f_new == nullptr || f_free == nullptr || f_init == nullptr || f_run == nullptr) {
        TS_LOG_ERROR << "f_new, f_free, f_init and f_run can't be nullptr" << eject;
    }
    std::string cpp_device(device);
    std::string cpp_op(op);
    auto creator = [=]() -> Operator::shared {
        return std::make_shared<APIPluginOperator>(cpp_device, cpp_op,
                f_new, f_free,
                f_init, f_infer, f_run);
    };
    OperatorCreator::Register(device, op, creator);
    TRY_TAIL
}

ts_Tensor *ts_OperatorParams_get(const ts_OperatorParams *dict, const char *param) {
    TRY_HEAD
    std::string cpp_param(param);
    if (dict->op->has(cpp_param)) {
        std::unique_ptr<ts_Tensor> value(new ts_Tensor(dict->op->get(cpp_param)));
        return value.release();
    }
    RETURN_OR_CATCH(nullptr, nullptr);
}