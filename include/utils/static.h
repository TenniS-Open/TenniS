//
// Created by lby on 2018/3/12.
//

#ifndef TENSORSTACK_UTILS_STATIC_H
#define TENSORSTACK_UTILS_STATIC_H

#include <utility>

namespace ts {
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

#endif //TENSORSTACK_UTILS_STATIC_H
