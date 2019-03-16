//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_API_ERRNO_H
#define TENSORSTACK_API_ERRNO_H

#include <string>

namespace ts {
    namespace api {
        extern thread_local std::string _thread_local_last_error_message;
    }
}

#endif //TENSORSTACK_API_ERRNO_H
