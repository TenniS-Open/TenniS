//
// Created by kier on 2019-04-14.
//

#include "utils/ctxmgr_lite.h"

#include <sstream>

namespace ts {
    static inline std::string no_lite_build_message(const std::thread::id &id) {
        std::ostringstream oss;
        oss << "Empty context in thread: " << id;
        return oss.str();
    }

    static inline std::string no_lite_build_message(const std::string &name, const std::thread::id &id) {
        std::ostringstream oss;
        oss << "Empty context: <" << name << "> in thread: " << id;
        return oss.str();
    }

    NoLiteContextException::NoLiteContextException()
            : NoLiteContextException(std::this_thread::get_id()) {
    }

    NoLiteContextException::NoLiteContextException(const std::thread::id &id)
            : Exception(no_lite_build_message(id)), m_thread_id(id) {
    }

    NoLiteContextException::NoLiteContextException(const std::string &name)
            : NoLiteContextException(name, std::this_thread::get_id()) {
    }

    NoLiteContextException::NoLiteContextException(const std::string &name, const std::thread::id &id)
            : Exception(no_lite_build_message(name, id)), m_thread_id(id) {
    }
}

