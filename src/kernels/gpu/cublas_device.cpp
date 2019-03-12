#include "kernels/gpu/cublas_device.h"

#include <kernels/gpu/cublas_device.h>

#include "global/memory_device.h"
#include "utils/static.h"
#include "kernels/gpu/memory_gpu.h"

namespace ts {
    void DeviceCuBLASAdminFunction(DeviceHandle **handle, int device_id, DeviceAdmin::Action action)
    {
        switch (action)
        {
        case ts::DeviceAdmin::INITIALIZATION:
        {
            auto cublas_device_handle = new CublasDevice(device_id);
            *handle = reinterpret_cast<DeviceHandle*>(cublas_device_handle);
            break;
        }
        case ts::DeviceAdmin::FINALIZATION:
        {
            delete reinterpret_cast<CublasDevice*>(*handle);
            break;
        }
        default:
            break;
        }
    }
}

TS_STATIC_ACTION(ts::HardAllocator::Register, ts::CUBLAS, ts::gpu_allocator)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::CUBLAS, ts::CUBLAS, ts::gpu2gpu_converter)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::CUBLAS, ts::CPU, ts::cpu2gpu_converter)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::CPU, ts::CUBLAS, ts::gpu2cpu_converter)
TS_STATIC_ACTION(ts::ComputingMemory::Register, ts::CUBLAS, ts::GPU)
TS_STATIC_ACTION(ts::DeviceAdmin::Register, ts::CUBLAS, ts::DeviceCuBLASAdminFunction)
