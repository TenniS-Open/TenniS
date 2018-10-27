//
// Created by kier on 2018/10/26.
//

#include "utils/ctxmgr.h"

namespace ts {
    TS_THREAD_LOCAL std::unordered_map<std::type_index, __thread_local_context>  __thread_local_type_context;
}
