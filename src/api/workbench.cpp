//
// Created by kier on 2019/3/16.
//

#include "declare_workbench.h"
#include "declare_module.h"
#include "declare_tensor.h"
#include "declare_image_filter.h"

using namespace ts;

ts_Workbench *ts_Workbench_Load(ts_Module *module, const ts_Device *device) {
    TRY_HEAD
    if (!module) throw Exception("NullPointerException: @param: 1");
    if (!device) throw Exception("NullPointerException: @param: 2");
    std::unique_ptr<ts_Workbench> workbench(new ts_Workbench(
            Workbench::Load(module->pointer, ComputingDevice(device->type, device->id))));
    RETURN_OR_CATCH(workbench.release(), nullptr)
}

void ts_free_Workbench(const ts_Workbench *workbench) {
    TRY_HEAD
    delete workbench;
    TRY_TAIL
}

ts_Workbench *ts_Workbench_clone(ts_Workbench *workbench) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Workbench> dolly(new ts_Workbench((*workbench)->clone()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Workbench_input(ts_Workbench *workbench, int32_t i, const ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    (*workbench)->input(i, **tensor);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_input_by_name(ts_Workbench *workbench, const char *name, const ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!name) throw Exception("NullPointerException: @param: 2");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    (*workbench)->input(name, **tensor);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_run(ts_Workbench *workbench) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    (*workbench)->run();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_output(ts_Workbench *workbench, int32_t i, ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    **tensor = (*workbench)->output(i);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_output_by_name(ts_Workbench *workbench, const char *name, ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!name) throw Exception("NullPointerException: @param: 2");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    **tensor = (*workbench)->output(name);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_set_computing_thread_number(ts_Workbench *workbench, int32_t number) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    (*workbench)->runtime().set_computing_thread_number(number);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_bind_filter(ts_Workbench *workbench, int32_t i, ts_ImageFilter *filter) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!filter) throw Exception("NullPointerException: @param: 3");
        (*workbench)->bind_filter(i, filter->pointer);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_bind_filter_by_name(ts_Workbench *workbench, const char *name, ts_ImageFilter *filter) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!name) throw Exception("NullPointerException: @param: 2");
        if (!filter) throw Exception("NullPointerException: @param: 3");
        (*workbench)->bind_filter(name, filter->pointer);
    RETURN_OR_CATCH(ts_true, ts_false)
}
