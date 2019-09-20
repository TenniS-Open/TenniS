//
// Created by kier on 19-4-28.
//

#include <runtime/workbench.h>
#include <runtime/image_filter.h>
#include <module/module.h>
#include <core/tensor_builder.h>
#include <board/hook.h>
#include <board/time_log.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <board/dump.h>

static inline void diff(const ts::Tensor &x, const ts::Tensor &y, float &max, float &avg) {
    auto float_x = ts::tensor::cast(ts::FLOAT32, x);
    auto float_y = ts::tensor::cast(ts::FLOAT32, y);
    auto count = std::min(x.count(), y.count());
    auto float_x_data = float_x.data<float>();
    auto float_y_data = float_y.data<float>();
    float local_max = 0;
    float local_sum = 0;
    for (int i = 0; i < count; ++i) {
        auto diff = std::fabs(float_x_data[i] - float_y_data[i]);
        local_sum += diff;
        if (local_max < diff) local_max = diff;
    }
    max = local_max;
    avg = local_sum / count;
}

int main() {
    using namespace ts;

    std::string gt_root = "/home/kier/git/TensorStack/python/test/holiday";
    std::string model = "/home/kier/git/TensorStack/bin/fas/SeetaAntiSpoofing.plg.1.0.m01d29.tsm";
    std::string input_t = gt_root + "/input.t";

    ComputingDevice device(CPU, 0);

    Hook hook;
    ctx::bind<Hook>_bind_hook(hook);

    auto module = Module::Load(model);
    auto bench = Workbench::Load(module, device);
    bench->setup_context();

    auto tensor = tensor::load(input_t);

    bench->input(0, tensor);


    hook.before_run([&](const Hook::StructBeforeRun &info) {
    });

    float max, avg;
    hook.after_run([&](const Hook::StructAfterRun &info) {
        std::string name = info.op->name();
        std::string op = info.op->op();

        auto fixed_name = name;
        for (auto &ch : fixed_name) {
            if (ch == '/' || ch == '\\') {
                ch = '#';
            }
        }

        auto wanted = tensor::load(gt_root + "/" + fixed_name + ".t");
        if (wanted.empty()) return;
        auto got = *info.stack->index(0);

        diff(wanted, got, max, avg);

        TS_LOG_INFO << "hook (" << name << ") " << op << ": " << wanted.proto() << " vs. " << got.proto() << ", max=" << max << ", avg=" << avg;
    });
    bench->run();

    return 0;
}