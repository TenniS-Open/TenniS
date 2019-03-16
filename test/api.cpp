//
// Created by kier on 2019/1/27.
//

#include <api/cpp/tensorstack.h>

#include <iostream>

int main() {
    auto abc = ts::api::tensor::from("abc");
    std::cout << ts::api::tensor::to_string(abc)<< std::endl;

    ts::api::Device device = {"cpu", 0};
    // load module
    auto module = ts::api::Module::Load("/Users/seetadev/Documents/SDK/CLion/TensorStack/bin/test.module.txt", TS_BINARY);

    // compile to workbench
    auto workbench = ts::api::Workbench::Load(module, device);

    // do compiler filter
    auto filter = ts::api::ImageFilter(device);

    // ts_Workbench_bind_filter(workbench, 0, filter);
    auto input_a = ts::api::tensor::build(TS_FLOAT32, {1}, {1});
    auto input_b = ts::api::tensor::build(TS_FLOAT32, {1}, {2});

    workbench.input(0, input_a);
    workbench.input(1, input_b);

    workbench.run();

    auto output = workbench.output(0);

    output.sync_cpu();  // sync number to cpu
    output = ts::api::tensor::cast(TS_FLOAT32, output);

    auto output_shape = output.sizes();

    std::cout << "output shape: [";
    for (int i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "output: " << output.data<float>()[0] << std::endl;

    return 0;
}