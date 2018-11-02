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

        /**
         * pointing to device operating self-defined structure
         * not using in out scope
         */
        DeviceHandle *handle = nullptr;
    };
}


#endif //TENSORSTACK_DEVICE_CONTEXT_H
