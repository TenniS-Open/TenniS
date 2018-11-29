//
// Created by kier on 2018/11/5.
//

#ifndef TENSORSTACK_UTILS_LOG_H
#define TENSORSTACK_UTILS_LOG_H



#include <iostream>
#include <sstream>
#include <cstring>

#include "except.h"
#include "box.h"

namespace ts {

    class EjectionException : public Exception {
    public:
        EjectionException() : Exception() {}
        explicit EjectionException(const std::string &message) : Exception(message) {}
    };

    enum LogLevel {
        LOG_NONE = 0,
        LOG_DEBUG = 1,
        LOG_STATUS = 2,
        LOG_INFO = 3,
        LOG_ERROR = 4,
        LOG_FATAL = 5,
    };

    inline std::string LogString(LogLevel level) {
        switch (level) {
            default:
                return "[Unknown]";
            case LOG_NONE:
                return "";
            case LOG_DEBUG:
                return "[DEBUG]";
            case LOG_STATUS:
                return "[STATUS]";
            case LOG_INFO:
                return "[INFO]";
            case LOG_ERROR:
                return "[ERROR]";
            case LOG_FATAL:
                return "[FATAL]";
        }
    }

    LogLevel GlobalLogLevel(LogLevel level);

    LogLevel GlobalLogLevel();

    class LogStream {
    public:
        using self = LogStream;

        explicit LogStream(LogLevel level, std::ostream &log = std::cout)
                : m_level(level), m_log(log) {
        }

        LogStream(const self &other) = delete;

        self &operator=(const self &other) = delete;

        ~LogStream() {
            flush();
        }

        const std::string message() const {
            return m_buffer.str();
        }

        template<typename T>
        self &operator()(const T &message) {
            if (m_level == LOG_NONE) return *this;
            if (m_level >= GlobalLogLevel()) {
                m_buffer << message;
            }
            return *this;
        }

        template<typename T>
        self &operator<<(const T &message) {
            return operator()(message);
        }

        using Method = self &(self &);

        self &operator<<(Method method) {
            if (m_level == LOG_NONE) return *this;
            if (m_level >= GlobalLogLevel()) {
                return method(*this);
            }
            return *this;
        }

        void flush() {
            if (m_level == LOG_NONE) return;
            if (m_level >= GlobalLogLevel()) {
                auto msg = m_buffer.str();
                m_buffer.str("");
                m_buffer << LogString(m_level) << ": " << msg << std::endl;
                m_log << m_buffer.str();
            }
            m_level = LOG_NONE;
            m_buffer.str("");
            m_log.flush();
        }

        LogLevel level() const { return m_level; }

    private:
        LogLevel m_level;
        std::ostringstream m_buffer;
        std::ostream &m_log;
    };

    inline LogStream &fatal(LogStream &log) {
        const auto msg = log.message();
        log.flush();
        std::exit(-1);
    }

    inline LogStream &eject(LogStream &log) {
        const auto msg = log.message();
        log.flush();
        throw EjectionException(msg);
    }
}

#ifdef TS_SOLUTION_DIR
#define TS_LOCAL_FILE ( \
    std::strlen(TS_SOLUTION_DIR) + 1 < std::strlen(__FILE__) \
    ? ((const char *)(__FILE__ + std::strlen(TS_SOLUTION_DIR) + 1)) \
    : ((const char *)(__FILE__)) \
    )
#else
#define TS_LOCAL_FILE (ts::Split(__FILE__, R"(/\)").back())
#endif

#define TS_LOG(level) (ts::LogStream(level))("[")(TS_LOCAL_FILE)(":")(__LINE__)("]: ")
#define TS_TIME(level) (ts::LogStream(level))("[")(ts::now_time())("]: ")
#define TS_LOG_TIME(level) (ts::LogStream(level))("[")(TS_LOCAL_FILE)(":")(__LINE__)("][")(ts::now_time())("]: ")

#define TS_LOG_DEBUG TS_LOG(ts::LOG_DEBUG)
#define TS_LOG_STATUS TS_LOG(ts::LOG_STATUS)
#define TS_LOG_INFO TS_LOG(ts::LOG_INFO)
#define TS_LOG_ERROR TS_LOG(ts::LOG_ERROR)
#define TS_LOG_FATAL TS_LOG(ts::LOG_FATAL)

#endif //TENSORSTACK_LOG_H
