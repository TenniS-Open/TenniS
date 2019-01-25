//
// Created by kier on 2019/1/25.
//

#ifndef TENSORSTACK_RUNTIME_IMAGE_FILTER_H
#define TENSORSTACK_RUNTIME_IMAGE_FILTER_H

#include <vector>
#include "core/tensor.h"
#include "utils/implement.h"

namespace ts {
    class ImageFilter {
    public:
        using self = ImageFilter;

        using shared = std::shared_ptr<self>;

        explicit ImageFilter(const ComputingDevice &device);

        ImageFilter(const self &) = delete;

        ImageFilter &operator=(const self &) = delete;

        void to_float();

        void scale(float f);

        void sub_mean(const std::vector<float> &mean);

        void resize(int width, int height);

        void center_crop(int width, int height);

        void channel_swap(const std::vector<int> &shuffle);

        void to_chw();

        /**
         * Clear all set processor
         */
        void clear();

        /**
         * Compile all processor
         */
        void compile();

        /**
         * Do ImageFilter
         * @param image Supporting Int8 and Float,
         *              Shape is [height, width, channels]
         * @return Converted image
         */
        Tensor run(const Tensor &image);

        shared clone() const;

    private:
        class Implement;
        Declare<Implement> m_impl;

        explicit ImageFilter(const Implement &other);

        std::string serial_name() const;
    };
}


#endif //TENSORSTACK_RUNTIME_IMAGE_FILTER_H
