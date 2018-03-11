//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_GLOBAL_DEVICE_H
#define TENSORSTACK_GLOBAL_DEVICE_H

#include <string>
#include <ostream>

namespace ts {
    /**
     * DeviceType: hardware includeing CPU, GPU or other predictable device
     */
    enum DeviceType {
        CPU = 0,    ///< CPU device
        GPU = 1,    ///< GPU device
        // Other memory m_type goes hear
    };

    /**
     * Device: Sepcific device
     */
    class Device {
    public:
        using self = Device;    ///< self class

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @param id Device type's id, 0, 1, or ...
         */
        Device(DeviceType type, int id) : m_type(type), m_id(id) {}

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @note Default id is 0
         */
        Device(DeviceType type) : self(type, 0) {}

        /**
         * Initialize device like CPU:0
         */
        Device() : self(CPU, 0) {}

        /**
         * Device type
         * @return Device type
         */
        const DeviceType type() const { return m_type; }

        /**
         * Device id
         * @return Device id
         */
        int id() const { return m_id; }

    private:
        DeviceType m_type = CPU;  ///< Hardware device @see Device
        int m_id = 0; ///< Device type's id, 0, 1, or ...
    };

    bool operator==(const Device &lhs, const Device &rhs);

    bool operator!=(const Device &lhs, const Device &rhs);

    bool operator<(const Device &lhs, const Device &rhs);

    bool operator>(const Device &lhs, const Device &rhs);

    bool operator<=(const Device &lhs, const Device &rhs);

    bool operator>=(const Device &lhs, const Device &rhs);
}

#endif //TENSORSTACK_GLOBAL_DEVICE_H
