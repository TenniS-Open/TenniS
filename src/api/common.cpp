//
// Created by kier on 2019/3/16.
//

#include "api/common.h"
#include "errno.h"

using namespace ts;
using namespace api;

const char *ts_last_error_message() {
    return _thread_local_last_error_message.c_str();
}