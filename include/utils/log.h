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

#undef NONE
#undef DEBUG
#undef STATUS
#undef INFO
#undef ERROR
#undef FATAL

namespace ts {

    enum LogLevel {
        NONE = 0,
        DEBUG = 1,
        STATUS = 2,
        INFO = 3,
        ERROR = 4,
        FATAL = 5,
    };

    extern LogLevel InnerGlobalLogLevel;

    inline LogLevel GlobalLogLevel(LogLevel level) {
        LogLevel pre_level = InnerGlobalLogLevel;
        InnerGlobalLogLevel = level;
        return pre_level;
    }

    inline LogLevel GlobalLogLevel() {
        return InnerGlobalLogLevel;
    }

    class LogCore {
    public:
        using self = LogCore;

        explicit  LogCore(std::ostream &stream)
                : m_stream(stream) {}

        LogCore()
                : m_stream(std::cout) {}

        virtual ~LogCore() = default;

        void add(const std::string &message) {
            m_buffer << message;
        }

        void commit() {
            m_buffer << std::endl;
            m_stream << m_buffer.str();
            m_buffer.str("");
        }

    private:
        std::ostream &m_stream;
        std::ostringstream m_buffer;
    };

    class LogCoreStream : public LogCore {
    public:
        using self = LogCoreStream;
        using supper = LogCore;

        explicit LogCoreStream(LogLevel level, std::ostream &stream)
                : supper(stream), m_level(level) {}

        explicit LogCoreStream(LogLevel level)
                : m_level(level) {}

        LogCoreStream()
                : m_level(NONE) {}

        ~LogCoreStream() {
            if (actived()) commit();
        }

        /**
         * @return true if actived
         */
        bool actived() const {
            return m_level >= GlobalLogLevel();
        }

        template <typename T>
        self &operator<<(const T &message) {
            if (!this->actived()) return *this;
            std::ostringstream oss;
            oss << message;
            this->add(oss.str());
            return *this;
        }

    private:
        LogLevel m_level;
    };

    class Log {
    public:
        Log(LogLevel level, std::ostream &log = std::cout)
                : m_level(level), m_log(log) {
        }

        ~Log() {
            flush();
        }

        const std::string message() const {
            return m_buffer.str();
        }

        template<typename T>
        Log &operator()(const T &message) {
            if (m_level >= InnerGlobalLogLevel) {
                m_buffer << message;
            }
            return *this;
        }

        template<typename T>
        Log &operator<<(const T &message) {
            return operator()(message);
        }

        using Method = Log &(Log &);

        Log &operator<<(Method method) {
            if (m_level >= InnerGlobalLogLevel) {
                return method(*this);
            }
            return *this;
        }

        void flush() {
            std::string level_str = "Unkown";
            switch (m_level) {
                case NONE:
                    return;
                case DEBUG:
                    level_str = "DEBUG";
                    break;
                case STATUS:
                    level_str = "STATUS";
                    break;
                case INFO:
                    level_str = "INFO";
                    break;
                case ERROR:
                    level_str = "ERROR";
                    break;
                case FATAL:
                    level_str = "FATAL";
                    break;
            }
            if (m_level >= InnerGlobalLogLevel) {
                auto msg = m_buffer.str();
                m_buffer.str("");
                m_buffer << level_str << ": " << msg << std::endl;
                m_log << m_buffer.str();
            }
            m_level = NONE;
            m_buffer.str("");
            m_log.flush();
        }

        LogLevel level() const { return m_level; }

    private:
        LogLevel m_level;
        std::ostringstream m_buffer;
        std::ostream &m_log;

        Log(const Log &other) = delete;

        Log &operator=(const Log &other) = delete;
    };

    inline Log &crash(Log &log) {
        if (log.level() == NONE) return log;

        const auto msg = log.message();
        log.flush();
        throw Exception(msg);
    }
}

#ifdef TS_SOLUTION_DIR
#define TS_LOCAL_FILE ( \
    std::strlen(TS_SOLUTION_DIR) + 1 < std::strlen(__FILE__) \
    ? ((const char *)(__FILE__ + std::strlen(TS_SOLUTION_DIR) + 1)) \
    : ((const char *)(__FILE__)) \
    )
#else
#define TS_LOCAL_FILE (orz::Split(__FILE__, R"(/\)").back())
#endif

#define TS_LOG(level) (orz::Log(level))("[")(TS_LOCAL_FILE)(":")(__LINE__)("]: ")
#define TS_TIME(level) (orz::Log(level))("[")(ts::now_time())("]: ")
#define TS_TIME_LOG(level) (orz::Log(level))("[")(ts::now_time())("][")(TS_LOCAL_FILE)(":")(__LINE__)("]: ")

#endif //TENSORSTACK_LOG_H
