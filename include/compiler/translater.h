//
// Created by kier on 2018/11/1.
//

#ifndef TENSORSTACK_COMPILER_TRANSLATER_H
#define TENSORSTACK_COMPILER_TRANSLATER_H

#include "module/module.h"
#include <core/device.h>

namespace ts {
    /**
     * translate Graph to TGraph
     * translate Graph from other framework to TS support Graph
     */
    class TS_DEBUG_API Translator {
    public:
        using self = Translator;

        explicit Translator(const ComputingDevice &device);

        Module::shared translate(const Module::shared& module) const;

    private:
        ComputingDevice m_device;
    };
}


#endif //TENSORSTACK_COMPILER_TRANSLATER_H
