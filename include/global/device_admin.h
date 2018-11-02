//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
#define TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H

#include "core/device_context.h"

#include <functional>

namespace ts {

    enum DeviceAction {
        DEVICE_INITIALIZATION,
        DEVICE_FINALIZATION,
    };

    /**
     * DeviceAdmin, initialize or finalize device shang xia wen
     */
    using DeviceAdmin = std::function<void(DeviceHandle **handle, int device_id, DeviceAction action)>;
}


#endif //TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
