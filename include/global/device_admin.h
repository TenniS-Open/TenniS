//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
#define TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H

#include "core/device_context.h"

#include <functional>

namespace ts {

    /**
     * Action on device
     */
    enum DeviceAction {
        DEVICE_INITIALIZATION,  ///< initialize a handle on given device
        DEVICE_FINALIZATION,    ///< finalize a handle on given device
    };

    /**
     * DeviceAdmin, initialize or finalize device shang xia wen
     */
    using DeviceAdmin = std::function<void(DeviceHandle **handle, int device_id, DeviceAction action)>;


    /**
     * Example of DeviceAdmin
     * @param handle Pointer to the pointer of handle, ready to be initialized of finalized, which controlled by `action`
     * @param device_id admin device id
     * @param action @sa DeviceAction
     */
    void DeviceAdminDeclaration(DeviceHandle **handle, int device_id, DeviceAction action);

    /**
     * Query device admin
     * @param device querying computing device
     * @return DeviceAdmin
     * @note supporting called by threads without calling @sa RegisterDeviceAdmin
     * @note the query device should be computing device
     */
    DeviceAdmin QueryDeviceAdmin(const DeviceType &device) TS_NOEXCEPT;

    /**
     * Register allocator for specific device type
     * @param device_type specific @sa DeviceType
     * @param allocator setting allocator
     * @note only can be called before running @sa QueryDeviceAdmin
     */
    void RegisterDeviceAdmin(const DeviceType &device_type, const DeviceAdmin &allocator) TS_NOEXCEPT;

    /**
     * No details for this API, so DO NOT call it
     */
    void ClearRegisteredDeviceAdmin();
}


#endif //TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
