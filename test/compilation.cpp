//
// Created by kier on 2019-06-10.
//

#include <module/module.h>

int main() {
    auto module = ts::Module::Load("/Users/seetadev/Documents/SDK/CLion/TensorStack/python/test/yolov3-tiny.tsm");
    ts::plot_graph(std::cout, module->outputs());
    module = ts::Module::Translate(module, ts::ComputingDevice(ts::CPU), "--float16");
    ts::plot_graph(std::cout, module->outputs());
}