//
// Created by kier on 2019-05-28.
//

#include <iostream>

#include "api/cpp/tennis.h"
#include "api/cpp/operator.h"

using namespace ts::api;

#include <functional>
#include <cassert>

class CustomOp : public Operator {
public:
    using self = CustomOp;
    float alpha;
    float beta;
    float gama;
    std::string binary;
    std::function<void(const float *, const float *, float *, int32_t)> op;

    void init(const OperatorParams &params, ts_OperatorContext *) override {
        alpha = tensor::cast(TS_FLOAT32, params.get("alpha")).data<float>(0);
        beta = tensor::to_float(params.get("beta"));
        gama = tensor::array::to_float(params.get("gama"))[0];
        binary = tensor::to_string(params.get("binary"));
        if (binary == "add")
            op = std::bind(
                    &CustomOp::add, this,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
        else if (binary == "sub")
            op = std::bind(
                    &CustomOp::sub, this,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
        else if (binary == "mul")
            op = std::bind(
                    &CustomOp::mul, this,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
        else if (binary == "div")
            op = std::bind(
                    &CustomOp::div, this,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
        else throw Exception("binary must be add, sub, mul or div, got " + binary);
    }

    std::vector<std::pair<DTYPE, Shape>> infer(const std::vector<Tensor> &args, ts_OperatorContext *) override {
        assert(args.size() == 2);
        return {std::make_pair(args[0].dtype(), args[0].sizes())};
    }

    void add(const float *a, const float *b, float *c, int32_t N) const {
        for (int32_t i = 0; i < N; ++i) c[i] = (alpha * a[i]) + (beta * b[i]) + gama;
    }

    void sub(const float *a, const float *b, float *c, int32_t N) const {
        for (int32_t i = 0; i < N; ++i) c[i] = (alpha * a[i]) - (beta * b[i]) + gama;
    }

    void mul(const float *a, const float *b, float *c, int32_t N) const {
        for (int32_t i = 0; i < N; ++i) c[i] = (alpha * a[i]) * (beta * b[i]) + gama;
    }

    void div(const float *a, const float *b, float *c, int32_t N) const {
        for (int32_t i = 0; i < N; ++i) c[i] = (alpha * a[i]) / (beta * b[i]) + gama;
    }

    std::vector<Tensor> run(const std::vector<Tensor> &args, ts_OperatorContext *) override {
        assert(args[0].dtype() == TS_FLOAT32);
        assert(args[1].dtype() == TS_FLOAT32);

        Tensor output(Tensor::InFlow::HOST, TS_FLOAT32, args[0].sizes());
        auto N = output.count();

        op(args[0].data<float>(), args[1].data<float>(), output.data<float>(), N);
        return {output};
    }
};


int main() {
    RegisterOperator<CustomOp>("cpu", "CustomOp");

    Device device("cpu", 0);
    std::string model = "customop.tsm";

    Workbench bench(device);
    bench.setup(bench.compile(Module::Load(model)));

    float value[] = {3, 4};
    Tensor a = tensor::build(TS_FLOAT32, 1, &value[0]);
    Tensor b = tensor::build(TS_FLOAT32, 1, &value[1]);
    bench.input(0, a);
    bench.input(1, b);

    bench.run();

    auto c = bench.output(0);

    std::cout << "a = " << a.data<float>(0)
              << ", b = " << b.data<float>(0)
              << ", c = " << c.data<float>(0)
              << std::endl;

    return 0;
}