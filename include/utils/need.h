//
// Created by Lby on 2017/10/9.
//

#ifndef TENSORSTACK_UTILS_NEED_H
#define TENSORSTACK_UTILS_NEED_H

#include "void_bind.h"

namespace ts {

    class need {
    public:

        template<typename FUNC>
        need(FUNC func) {
            task = void_bind(func);
        }

        template<typename FUNC, typename... Args>
        need(FUNC func, Args &&... args) {
            task = void_bind(func, std::forward<Args>(args)...);
        }

        ~need() { task(); }

    private:
        need(const need &that) = delete;

        need &operator=(const need &that) = delete;

        VoidOperator task;
    };
}

#endif //TENSORSTACK_UTILS_NEED_H
