//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_DEVICE_CONTEXT_H
#define TENSORSTACK_DEVICE_CONTEXT_H


#include <global/hard_converter.h>
#include <global/device_admin.h>

namespace ts {
    class DeviceHandle;
    class DeviceContext {
    public:
        using self = DeviceContext;
        using shared = std::shared_ptr<self>;

        DeviceContext() = default;

        DeviceContext(ComputingDevice computing_device);

        ~DeviceContext();

        DeviceContext(const self &) = delete;
        self &operator=(const self &) = delete;

        void initialize(ComputingDevice computing_device);
        void finalize();

        /**
         * pointing to device operating self-defined structure
         * not using in out scope
         */
        DeviceHandle *handle = nullptr;

        ComputingDevice computing_device;
        MemoryDevice memory_device;

    private:
        DeviceAdmin::function m_device_admin;
    };
}


#endif //TENSORSTACK_DEVICE_CONTEXT_H
