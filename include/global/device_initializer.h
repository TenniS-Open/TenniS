//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
#define TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H

#include <functional>

namespace ts {
    class DeviceContext;
    /**
     * DeviceInitializer, initialize device shang xia wen
     */
    using DeviceInitializer = std::function<void(DeviceContext *context, int device_id)>;
    /**
     * DeviceFinalizer, finalize device shang xia wen
     */
    using DeviceFinalizer = std::function<void(DeviceContext *context)>;
}


#endif //TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
