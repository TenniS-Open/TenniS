//
// Created by kier on 2018/10/26.
//

#include "utils/ctxmgr.h"

namespace ts {
    std::unordered_map<std::type_index, __thread_context> __global_thread_context;
}
