//
// Created by yang on 2020/2/26.
//

#ifndef TENSORSTACK_RUNTIME_SWITCHER_H
#define TENSORSTACK_RUNTIME_SWITCHER_H

#include "api/api.h"
#include "core/device.h"

class Importor;

namespace ts{

    class TS_DEBUG_API Switcher{
    public:
        using self = Switcher;
        using shared = std::shared_ptr<self>;

        Switcher();
        ~Switcher();

        void auto_switch(const ComputingDevice &device);

    private:
        void free();

    private:
        std::shared_ptr<Importor> m_importer;

        std::shared_ptr<ts_op_creator_map> m_creator_map;
        std::shared_ptr<ts_device_context> m_device_context;

    };
}

#endif //TENSORSTACK_RUNTIME_SWITCHER_H
