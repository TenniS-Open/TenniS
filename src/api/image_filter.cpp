//
// Created by kier on 2019/3/16.
//

#include "declare_image_filter.h"

using namespace ts;

ts_ImageFilter *ts_new_ImageFilter(const ts_Device *device) {
    TRY_HEAD
    //if (!device) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_ImageFilter> image_filter(
            device
            ? new ts_ImageFilter(ComputingDevice(device->type, device->id))
            : new ts_ImageFilter()
    );
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

ts_bool ts_ImageFilter_div_std(ts_ImageFilter *filter, const float *std, int32_t len) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    if (!std) throw Exception("NullPointerException: @param: 2");
    (*filter)->div_std(std::vector<float>(std, std + len));
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