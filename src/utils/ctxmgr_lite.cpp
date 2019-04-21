//
// Created by kier on 2019-04-14.
//

#include "utils/ctxmgr_lite.h"
#include "utils/platform.h"

#include <sstream>

namespace ts {
    static inline std::string no_lite_build_message(const std::thread::id &id) {
        std::ostringstream oss;
        oss << "Empty context in thread: " << id;
        return oss.str();
    }

    static inline std::string no_lite_build_message(const std::string &name, const std::thread::id &id) {
        std::ostringstream oss;
        oss << "Empty context:<" << classname(name) << "> in thread: " << id;
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

#if TS_PLATFORM_CC_GCC
    static bool is_number(char ch) {
        return ch >= '0' && ch <= '9';
    }

    static std::string parse_part(const char *str, const char **ret) {
        char *end = nullptr;
        auto num = std::strtol(str, &end, 10);
        if (str == end) return "";
        std::string parsed(end, end + num);
        *ret = end + num;
        return std::move(parsed);
    }

    static std::string parse_top(const char *str) {
        if (*str == '\0') return "";
        if (is_number(*str)) return parse_part(str, &str);
        if (*str != 'N') return "";
        ++str;
        std::string parsed;
        while (true) {
            if (*str == 'E') break;
            auto part = parse_part(str, &str);
            if (part.empty()) return "";
            if (!parsed.empty()) {
                parsed += "::";
            }
            parsed += part;
        }
        return parsed;
    }

    static std::string classname_gcc(const std::string &name) {
        auto parsed = parse_top(name.c_str());
        if (parsed.empty()) return name;
        return parsed;
    }
#endif

    std::string classname(const std::string &name) {
#if TS_PLATFORM_CC_MSVC
        return name;
#elif TS_PLATFORM_CC_MINGW
        return name;
#elif TS_PLATFORM_CC_GCC
        return classname_gcc(name);
#else
        return name;
#endif
    }
}

