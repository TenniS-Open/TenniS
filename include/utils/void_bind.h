//
// Created by lby on 2018/1/23.
//

#ifndef TENSORSTACK_UTILS__VOID_BIND_H
#define TENSORSTACK_UTILS__VOID_BIND_H

#include <functional>

namespace ts {

    using VoidOperator = std::function<void()>;

    template<typename Ret, typename FUNC>
    class _VoidOperatorBinder {
    public:
        static VoidOperator bind(FUNC func) { return [func]() -> void { func(); }; }
    };

    template<typename FUNC>
    class _VoidOperatorBinder<void, FUNC> {
    public:
        static VoidOperator bind(FUNC func) { return func; }
    };

    template<typename FUNC, typename... Args>
    static VoidOperator void_bind(FUNC func, Args &&... args) {
        auto inner_func = std::bind(func, std::forward<Args>(args)...);
        using Ret = decltype(inner_func());
        using RetOperator = _VoidOperatorBinder<Ret, decltype(inner_func)>;
        return RetOperator::bind(inner_func);
    }
}

#endif //TENSORSTACK_UTILS__VOID_BIND_H
