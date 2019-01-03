//
// Created by kier on 2018/11/2.
//

#include <core/device_context.h>

#include "core/device_context.h"
#include "global/memory_device.h"

#include "utils/ctxmgr_lite_support.h"

namespace ts {
    DeviceContext::~DeviceContext() {
        this->finalize();
    }

    void DeviceContext::initialize(ComputingDevice computing_device) {
        this->computing_device = computing_device;
        this->memory_device = ComputingMemory::Query(computing_device);
        this->m_device_admin = DeviceAdmin::Query(computing_device.type());

        finalize();
        if (m_device_admin != nullptr) {
            m_device_admin(&this->handle, computing_device.id(), DeviceAdmin::INITIALIZATION);
        }
    }

    void DeviceContext::finalize() {
        if (m_device_admin != nullptr && this->handle != nullptr) {
            m_device_admin(&this->handle, computing_device.id(), DeviceAdmin::FINALIZATION);
            this->handle = nullptr;
        }
    }

    DeviceContext::DeviceContext(ComputingDevice computing_device) {
        this->initialize(computing_device);
    }
}

TS_LITE_CONTEXT(ts::DeviceContext)
