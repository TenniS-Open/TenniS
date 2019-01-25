//
// Created by kier on 2019/1/25.
//

#ifndef TENSORSTACK_RUNTIME_IMAGE_FILTER_H
#define TENSORSTACK_RUNTIME_IMAGE_FILTER_H

#include <vector>
#include "core/tensor.h"
#include "workbench.h"

namespace ts {
    class ImageFilter {
    public:
        using self = ImageFilter;

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

    private:
        ComputingDevice m_computing_device;
        Workbench::shared m_workbench;
        Graph::shared m_graph;
        bool m_compiled = false;

        std::string serial_name() const;
    };
}


#endif //TENSORSTACK_RUNTIME_IMAGE_FILTER_H
