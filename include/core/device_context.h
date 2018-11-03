//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_DEVICE_CONTEXT_H
#define TENSORSTACK_DEVICE_CONTEXT_H


#include <global/hard_converter.h>

namespace ts {
    class DeviceHandle;
    class DeviceContext {
    public:
        using self = DeviceContext;
        using shared = std::shared_ptr<self>;

        DeviceContext(const self &) = delete;
        self &operator=(const self &) = delete;

        /**
         * pointing to device operating self-defined structure
         * not using in out scope
         */
        DeviceHandle *handle = nullptr;
    };
}


#endif //TENSORSTACK_DEVICE_CONTEXT_H
