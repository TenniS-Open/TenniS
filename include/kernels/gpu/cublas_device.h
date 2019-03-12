#ifndef TENSORSTACK_GLOBAL_CUBLAS_DEVICE_H
#define TENSORSTACK_GLOBAL_CUBLAS_DEVICE_H

#include "global/device_admin.h"
#include "core/device.h"
#include "utils/log.h"
#include "cublas_v2.h"

#include <cublas_v2.h>

namespace ts {

    class CublasDevice : public Device {
    public:
        using self = CublasDevice;
        using supper = Device;

        CublasDevice(int id) : supper(CUBLAS, id) { 
            if(cublasCreate(&m_handle) != cudaSuccess)
                TS_LOG_ERROR << "The cublasHandle initialize failed " << eject;
        }

        CublasDevice() : supper(CUBLAS) { 
            if (cublasCreate(&m_handle) != cudaSuccess)
                TS_LOG_ERROR << "The cublasHandle initialize failed " << eject;
        }

        ~CublasDevice() { cublasDestroy(m_handle); }

        cublasHandle_t get() { return std::move(m_handle); }
    private:
        cublasHandle_t m_handle;
    };

    void DeviceCuBLASAdminFunction(DeviceHandle **handle, int device_id, DeviceAdmin::Action action);
}

#endif //TENSORSTACK_GLOBAL_CUBLAS_DEVICE_H