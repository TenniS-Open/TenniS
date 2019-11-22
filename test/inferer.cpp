//
// Created by kier on 2019/11/22.
//

#include "runtime/inferer.h"
#include "module/module.h"
#include "core/tensor_builder.h"

int main() {
    std::string module_path = "/home/kier/git/TensorStack/python/test/caffe.tsm";

    auto module = ts::Module::Load(module_path);

    auto input = module->input(0);
    if (!input->has("#shape"))
        input->set("#shape", ts::tensor::build(ts::INT32, {-1, 256, 256, 3}));

    auto output = module->output(0);

    auto output_proto = ts::infer(output);

    TS_LOG_INFO << output_proto;

    return 0;
}