//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_CORE_DEVICE_H
#define TENSORSTACK_CORE_DEVICE_H

#include "utils/api.h"
#include "utils/except.h"

#include <sstream>
#include <string>
#include <ostream>
#include <memory>

namespace ts {
    /**
     * DeviceType: hardware includeing CPU, GPU or other predictable device
     */
    using DeviceType = std::string;
    static const char *CPU = "cpu";
    static const char *GPU = "gpu";
    static const char *EIGEN = "eigen";
    static const char *BLAS = "blas";
    static const char *CUDNN = "cudnn";
    static const char *CUBLAS = "cublas";

    /**
     * Device: Sepcific device
     */
    class Device {
    public:
        using self = Device;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @param id Device type's id, 0, 1, or ...
         */
        Device(const DeviceType &type, int id) : m_type(type), m_id(id) {}

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @note Default id is 0
         */
        Device(const DeviceType &type) : self(type, 0) {}

        /**
         * Initialize device like CPU:0
         */
        Device() : self(CPU, 0) {}

        /**
         * Device type
         * @return Device type
         */
        const DeviceType &type() const { return m_type; }

        /**
         * Device id
         * @return Device id
         */
        int id() const { return m_id; }

        /**
         * return repr string
         * @return repr string
         */
        const std::string repr() const { return m_type + ":" + std::to_string(m_id); }

        /**
         * return string show the content
         * @return string
         */
        const std::string str() const { return m_type + ":" + std::to_string(m_id); }

    private:
        DeviceType m_type = CPU;  ///< Hardware device @see Device
        int m_id = 0; ///< Device type's id, 0, 1, or ...
    };

    inline std::ostream &operator<<(std::ostream &out, const Device &device) {
        TS_UNUSED(CPU);
        TS_UNUSED(GPU);
        TS_UNUSED(EIGEN);
        TS_UNUSED(BLAS);
        TS_UNUSED(CUDNN);
        TS_UNUSED(CUBLAS);
        return out << device.str();
    }

    bool operator==(const Device &lhs, const Device &rhs);

    bool operator!=(const Device &lhs, const Device &rhs);

    bool operator<(const Device &lhs, const Device &rhs);

    bool operator>(const Device &lhs, const Device &rhs);

    bool operator<=(const Device &lhs, const Device &rhs);

    bool operator>=(const Device &lhs, const Device &rhs);

    class DeviceMismatchException : public Exception {
    public:
        explicit DeviceMismatchException(const Device &needed, const Device &given);

        static std::string DeviceMismatchMessage(const Device &needed, const Device &given);

        const Device &needed() const { return m_needed; }

        const Device &given() const { return m_given; }

    private:
        Device m_needed;
        Device m_given;
    };

    class MemoryDevice : public Device {
    public:
        using self = MemoryDevice;
        using supper = Device;

        MemoryDevice(const DeviceType &type, int id) : supper(type, id) {}

        MemoryDevice(const DeviceType &type) : supper(type) {}

        MemoryDevice() : supper() {}

    };
    class ComputingDevice : public Device {
    public:
        using self = ComputingDevice;
        using supper = Device;

        ComputingDevice(const DeviceType &type, int id) : supper(type, id) {}

        ComputingDevice(const DeviceType &type) : supper(type) {}

        ComputingDevice() : supper() {}
    };

}

namespace std {
    template<>
    struct hash<ts::Device> {
        std::size_t operator()(const ts::Device &key) const {
            using std::size_t;
            using std::hash;

            return hash<ts::DeviceType>()(key.type())
                   ^ hash<int>()(key.id());
        }
    };
}

#endif //TENSORSTACK_CORE_DEVICE_H
