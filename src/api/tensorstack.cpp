//
// Created by kier on 2019/1/26.
//

#include "api/tensorstack.h"

#include <string>
#include <api/tensorstack.h>


#include "global/setup.h"
#include "utils/except.h"
#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "module/module.h"
#include "runtime/workbench.h"
#include "runtime/image_filter.h"

static thread_local std::string _thread_local_last_error_message;

using namespace ts;

#define DECLARE_API_TYPE(API_TYPE, TS_TYPE) \
struct API_TYPE { \
    using self = API_TYPE; \
    template <typename... Args> \
    explicit API_TYPE(Args &&...args) { \
        this->pointer = std::make_shared<TS_TYPE>(std::forward<Args>(args)...); \
    } \
    explicit API_TYPE(std::shared_ptr<TS_TYPE> pointer) : pointer(std::move(pointer)) {} \
    std::shared_ptr<TS_TYPE> pointer; \
    const TS_TYPE *operator->() const { return pointer.get(); } \
    TS_TYPE *operator->() { return pointer.get(); } \
    const TS_TYPE &operator*() const { return *pointer; } \
    TS_TYPE &operator*() { return *pointer; } \
    const TS_TYPE *get() const { return pointer.get(); } \
    TS_TYPE *get() { return pointer.get(); } \
};

DECLARE_API_TYPE(ts_Tensor, Tensor)
DECLARE_API_TYPE(ts_Module, Module)
DECLARE_API_TYPE(ts_Workbench, Workbench)
DECLARE_API_TYPE(ts_ImageFilter, ImageFilter)

#define ts_false 0
#define ts_true 1

#define TRY_HEAD \
_thread_local_last_error_message.clear(); \
try {

#define RETURN_OR_CATCH(ret, cat) \
return ret; \
} catch (const Exception &e) { \
_thread_local_last_error_message = e.what(); \
return cat; \
}

#define TRY_TAIL \
} catch (const Exception &) { \
}

ts_bool ts_setup() {
    TRY_HEAD
        setup();
    RETURN_OR_CATCH(ts_true, ts_false);
}

const char *ts_last_error_message() {
    return _thread_local_last_error_message.c_str();
}

ts_Tensor *ts_new_Tensor(int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data) {
    TRY_HEAD
        if (shape == nullptr) shape_len = 0;
        std::unique_ptr<ts_Tensor> tensor(new ts_Tensor());
    
        if (data == nullptr) {
            **tensor = Tensor(DTYPE(dtype), Shape(shape, shape + shape_len));
            return tensor.release();
        }

        switch (dtype) {
            default:
                TS_LOG_ERROR << "Not support dtype: " << dtype << eject;
                break;
#define DECLARE_TENSOR_BUILD(api_dtype, type) \
        case api_dtype: \
            **tensor = tensor::build(DTYPE(dtype), Shape(shape, shape + shape_len), \
                    reinterpret_cast<const type*>(data)); \
            break;

            DECLARE_TENSOR_BUILD(TS_INT8, int8_t)
            DECLARE_TENSOR_BUILD(TS_UINT8, uint8_t)
            DECLARE_TENSOR_BUILD(TS_INT16, int16_t)
            DECLARE_TENSOR_BUILD(TS_UINT16, uint16_t)
            DECLARE_TENSOR_BUILD(TS_INT32, int32_t)
            DECLARE_TENSOR_BUILD(TS_UINT32, uint32_t)
            DECLARE_TENSOR_BUILD(TS_INT64, int64_t)
            DECLARE_TENSOR_BUILD(TS_UINT64, uint64_t)
            DECLARE_TENSOR_BUILD(TS_FLOAT32, float)
            DECLARE_TENSOR_BUILD(TS_FLOAT64, double)

#undef DECLARE_TENSOR_BUILD
        }

    RETURN_OR_CATCH(tensor.release(), nullptr)
}

void ts_free_Tensor(const ts_Tensor *tensor) {
    TRY_HEAD
        delete tensor;
    TRY_TAIL
}

const int32_t *ts_Tensor_shape(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        const int32_t *shape = (*tensor)->sizes().data();
    RETURN_OR_CATCH(shape, nullptr)
}

int32_t ts_Tensor_shape_size(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        auto size = int32_t((*tensor)->dims());
    RETURN_OR_CATCH(size, 0)
}

ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        auto dtype = ts_DTYPE((*tensor)->proto().dtype());
    RETURN_OR_CATCH(dtype, TS_VOID)
}

void *ts_Tensor_data(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        auto data = (*tensor)->data();
    RETURN_OR_CATCH(data, nullptr)
}

ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor((*tensor)->clone_shared()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        (*tensor)->sync(MemoryDevice(CPU));
    RETURN_OR_CATCH(true, false)
}

ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format) {
    TRY_HEAD
        if (!filename) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Module> module(new ts_Module(
                Module::Load(filename, Module::SerializationFormat(format))));
    RETURN_OR_CATCH(module.release(), nullptr)
}

void ts_free_Module(const ts_Module *module) {
    TRY_HEAD
        delete module;
    TRY_TAIL
}

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

ts_ImageFilter *ts_new_ImageFilter(const ts_Device *device) {
    TRY_HEAD
        if (!device) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_ImageFilter> image_filter(new ts_ImageFilter(ComputingDevice(device->type, device->id)));
    RETURN_OR_CATCH(image_filter.release(), nullptr)
}

void ts_free_ImageFilter(const ts_ImageFilter *filter) {
    TRY_HEAD
        delete filter;
    TRY_TAIL
}

ts_bool ts_ImageFilter_clear(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->clear();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_compile(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->compile();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_to_float(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->to_float();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_scale(ts_ImageFilter *filter, float f) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->scale(f);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_sub_mean(ts_ImageFilter *filter, const float *mean, int32_t len) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        if (!mean) throw Exception("NullPointerException: @param: 2");
        (*filter)->sub_mean(std::vector<float>(mean, mean + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_resize(ts_ImageFilter *filter, int width, int height) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->resize(width, height);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_center_crop(ts_ImageFilter *filter, int width, int height) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->center_crop(width, height);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_channel_swap(ts_ImageFilter *filter, const int *shuffle, int32_t len) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        if (!shuffle) throw Exception("NullPointerException: @param: 2");
        (*filter)->channel_swap(std::vector<int>(shuffle, shuffle + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_to_chw(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->to_chw();
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








