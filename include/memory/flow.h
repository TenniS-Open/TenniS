//
// Created by kier on 2019/1/8.
//

#ifndef TENSORSTACK_MEMORY_FLOW_H
#define TENSORSTACK_MEMORY_FLOW_H

#include <utils/implement.h>
#include "core/controller.h"


namespace ts {
    class QueuedStackMemoryController : public MemoryController {
    public:
        using self = QueuedStackMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        explicit QueuedStackMemoryController(const MemoryDevice &device);

        Memory alloc(size_t size) override;
    };


    class VatMemoryController : public MemoryController {
    public:
        using self = VatMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        explicit VatMemoryController(const MemoryDevice &device);

        Memory alloc(size_t size) override;

    private:
        class Implement;
        Declare<Implement> m_impl;
    };


    class StackMemoryController : public MemoryController {
    public:
        using self = VatMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        explicit StackMemoryController(const MemoryDevice &device);

        Memory alloc(size_t size) override;

    private:
        class Implement;
        Declare<Implement> m_impl;
    };


    using FlowMemoryController = VatMemoryController;
}


#endif //TENSORSTACK_MEMORY_FLOW_H
