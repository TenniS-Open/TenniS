//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_GLOBAL_ALLOCATOR_H
#define TENSORSTACK_GLOBAL_ALLOCATOR_H

#include "device.h"
#include <functional>

namespace ts {
    /**
     * Memory allocator type, allocate memory from specific device
     * TODO: may need this API support same device but different id memory allocate
     */
    using HardAllocator = std::function<void *(size_t, void *)>;

    /**
     * Query memory allocator
     * @param device querying device
     * @return allocator
     */
    HardAllocator QueryAllocator(const Device &device) noexcept;

    /**
     * Register allocator for specific device type
     * @param device_type specific @sa DeviceType
     * @param allocator setting allocator
     */
    void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) noexcept;

    /**
     * StaticAction: for supporting static initialization
     */
    class StaticAction {
    public:
        template <typename FUNC, typename... Args>
        explicit StaticAction(FUNC func, Args&&... args) noexcept {
            func(std::forward<Args>(args)...);
        }
    };
}

#define _ts_concat_name_core(x,y) (x##y)

#define _ts_concat_name(x, y) _ts_concat_name_core(x,y)

/**
 * generate an serial name by line
 */
#define ts_serial_name(x) _ts_concat_name(x, __LINE__)

/**
 * Static action
 */
#define TS_STATIC_ACTION(func, args...) \
    namespace \
    { \
         ts::StaticAction ts_serial_name(_ts_static_action_)(func, ##args); \
    }


#endif //TENSORSTACK_GLOBAL_ALLOCATOR_H
