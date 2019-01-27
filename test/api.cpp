//
// Created by kier on 2019/1/27.
//

#include <api/tensorstack.h>

#include <iostream>

int main() {
    ts_Device device = {"cpu", 0};
    // load module
    auto module = ts_Module_Load("/Users/seetadev/Documents/SDK/CLion/TensorStack/python/test/test.module.txt", TS_BINARY);
    if (module == nullptr) {
        std::cerr << ts_last_error_message() << std::endl;
        return 1;
    }
    // compile to workbench
    auto workbench = ts_Workbench_Load(module, &device);
    if (workbench == nullptr) {
        std::cerr << ts_last_error_message() << std::endl;
        return 1;
    }
    // ! ============= !
    ts_free_Module(module); // the module can be free after load

    // do compiler filter
    auto filter = ts_new_ImageFilter(&device);
    if (filter == nullptr) {
        std::cerr << ts_last_error_message() << std::endl;
        return 1;
    }
    // ts_Workbench_bind_filter(workbench, 0, filter);
    int32_t input_shape[] = {1};
    auto input_a = ts_new_Tensor(input_shape, 1, TS_FLOAT32, nullptr);
    auto input_b = ts_new_Tensor(input_shape, 1, TS_FLOAT32, nullptr);
    auto output = ts_new_Tensor(nullptr, 0, TS_VOID, nullptr);  // build empty tensor for output

    reinterpret_cast<float*>(ts_Tensor_data(input_a))[0] = 1;
    reinterpret_cast<float*>(ts_Tensor_data(input_b))[0] = 3;

    ts_Workbench_input(workbench, 0, input_a);
    ts_Workbench_input(workbench, 1, input_b);

    ts_Workbench_run(workbench);

    ts_Workbench_output(workbench, 0, output);

    ts_Tensor_sync_cpu(output); // sync to cpu, ready to write

    auto output_shape = ts_Tensor_shape(output);
    auto output_shape_size = ts_Tensor_shape_size(output);

    std::cout << "output shape: [";
    for (int i = 0; i < output_shape_size; ++i) {
        std::cout << output_shape[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "output: " << reinterpret_cast<float*>(ts_Tensor_data(output))[0] << std::endl;

    ts_free_Tensor(input_a);
    ts_free_Tensor(input_b);
    ts_free_Tensor(output);
    ts_free_Workbench(workbench);
    ts_free_ImageFilter(filter);

    return 0;
}