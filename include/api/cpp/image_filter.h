//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_API_CPP_IMAGE_FILTER_H
#define TENSORSTACK_API_CPP_IMAGE_FILTER_H

#include "../image_filter.h"

#include "except.h"
#include "device.h"

#include <string>
#include <vector>

namespace ts {
    namespace api {
        class ImageFilter {
        public:
            using self = ImageFilter;
            using raw = ts_ImageFilter;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            ImageFilter(const self &) = default;

            ImageFilter &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            ImageFilter() : self(Device()) {}

            ImageFilter(const Device &device) : self(device.get_raw()) {}

            ImageFilter(const ts_Device *device) : self(ts_new_ImageFilter(device)) {
                TS_API_AUTO_CHECK(m_impl != nullptr);
            }

            void clear() {
                TS_API_AUTO_CHECK(ts_ImageFilter_clear(m_impl.get()));
            }

            void compile() {
                TS_API_AUTO_CHECK(ts_ImageFilter_compile(m_impl.get()));
            }

            void to_float() {
                TS_API_AUTO_CHECK(ts_ImageFilter_to_float(m_impl.get()));
            }

            void scale(float scale) {
                TS_API_AUTO_CHECK(ts_ImageFilter_scale(m_impl.get(), scale));
            }

            void sub_mean(const std::vector<float> &mean) {
                TS_API_AUTO_CHECK(ts_ImageFilter_sub_mean(m_impl.get(), mean.data(), int(mean.size())));
            }

            void div_std(const std::vector<float> &std) {
                TS_API_AUTO_CHECK(ts_ImageFilter_div_std(m_impl.get(), std.data(), int(std.size())));
            }

            void resize(int width, int height) {
                TS_API_AUTO_CHECK(ts_ImageFilter_resize(m_impl.get(), width, height));
            }

            void resize(int width) {
                TS_API_AUTO_CHECK(ts_ImageFilter_resize_scalar(m_impl.get(), width));
            }

            void center_crop(int width, int height) {
                TS_API_AUTO_CHECK(ts_ImageFilter_center_crop(m_impl.get(), width, height));
            }

            void center_crop(int width) {
                TS_API_AUTO_CHECK(ts_ImageFilter_center_crop(m_impl.get(), width, width));
            }

            void channel_swap(const std::vector<int> &shuffle) {
                TS_API_AUTO_CHECK(ts_ImageFilter_channel_swap(m_impl.get(), shuffle.data(), int(shuffle.size())));
            }

            void to_chw() {
                TS_API_AUTO_CHECK(ts_ImageFilter_to_chw(m_impl.get()));
            }

        private:
            ImageFilter(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_ImageFilter); }

            shared_raw m_impl;
        };
    }
}

#endif //TENSORSTACK_API_CPP_IMAGE_FILTER_H
