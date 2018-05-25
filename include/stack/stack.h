//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_STACK_STACK_H
#define TENSORSTACK_STACK_STACK_H

#include <global/device.h>
#include <mem/controller.h>

namespace ts {

    class stack {
    public:
        stack();

        void push();

        void pop();

        void index();

    private:
        Device m_device;                          ///< running tensor device, compute on it
        MemoryController::shared m_controller;    ///< tensor memory backend
    };
}


#endif //TENSORSTACK_STACK_STACK_H
