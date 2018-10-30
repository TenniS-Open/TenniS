//
// Created by kier on 2018/10/26.
//

#ifndef TENSORSTACK_UTILS_CTXMGR_H
#define TENSORSTACK_UTILS_CTXMGR_H

#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <stack>
#include <thread>
#include <sstream>

#include "except.h"

#define TS_THREAD_LOCAL thread_local

namespace ts {

    class NoContextException : public Exception {
    public:
        NoContextException()
                : NoContextException(std::this_thread::get_id()) {
        }

        explicit NoContextException(const std::thread::id &id)
                : Exception(NoContextMessage(id)), m_thread_id(id) {
        }

    private:
        static std::string NoContextMessage(const std::thread::id &id) {
            std::ostringstream oss;
            oss << "Empty context in thread: " << id;
            return oss.str();
        }

        std::thread::id m_thread_id;
    };

    class __thread_local_context {
    public:
        using self = __thread_local_context;

        using context = void *;

        context swap(context ctx) {
            auto pre_ctx = this->m_ctx;
            this->m_ctx = ctx;
            return pre_ctx;
        }

        void set(context ctx) {
            this->m_ctx = ctx;
        }

        const context get() const {
            if (m_ctx == nullptr) throw NoContextException();
            return m_ctx;
        }

        context get() {
            return reinterpret_cast<context>(reinterpret_cast<const self *>(this)->get());
        }

        const context try_get() const { return this->m_ctx; }

        context try_get() { return this->m_ctx; }

    private:
        context m_ctx = nullptr;
    };

    extern TS_THREAD_LOCAL
    std::unordered_map<std::type_index, __thread_local_context> __thread_local_type_context;

    class __context {
    public:
        using self = __context;
        using context = void *;

        explicit __context(const std::type_index &type, context ctx) {
            auto &thread_local_context = __thread_local_type_context[type];
            this->m_context = &thread_local_context;
            this->m_now_ctx = ctx;
            this->m_pre_ctx = thread_local_context.swap(ctx);
        }

        ~__context() {
            this->m_context->set(this->m_pre_ctx);
        }

        static void set(const std::type_index &type, context ctx) {
            auto &thread_local_context = __thread_local_type_context[type];
            thread_local_context.set(ctx);
        }

        static context get(const std::type_index &type) {
            auto &thread_local_context = __thread_local_type_context[type];
            return thread_local_context.get();
        }

        static context try_get(const std::type_index &type) {
            auto &thread_local_context = __thread_local_type_context[type];
            return thread_local_context.try_get();
        }

        __context(const self &) = delete;

        self &operator=(const self &) = delete;

        context ctx() { return m_now_ctx; }

        const context ctx() const { return m_now_ctx; }

    private:
        context m_pre_ctx = nullptr;
        context m_now_ctx = nullptr;
        __thread_local_context *m_context = nullptr;
    };

    namespace ctx {
        template<typename T>
        class bind {
        public:
            using self = bind;

            explicit bind(T *ctx)
                    : m_ctx(std::type_index(typeid(T)), ctx) {
            }

            explicit bind(T &ctx_ref)
                    : bind(&ctx_ref) {
            }

            ~bind() = default;

            bind(const self &) = delete;

            self &operator=(const self &) = delete;

        private:
            __context m_ctx;
        };

        template<typename T>
        inline T *get() {
            return reinterpret_cast<T *>(__context::try_get(std::type_index(typeid(T))));
        }

        template<typename T>
        inline T *ptr() {
            return reinterpret_cast<T *>(__context::try_get(std::type_index(typeid(T))));
        }

        template<typename T>
        inline T &ref() {
            return *reinterpret_cast<T *>(__context::get(std::type_index(typeid(T))));
        }

        template<typename T, typename... Args>
        inline void initialize(Args &&...args) {
            auto ctx = new T(std::forward<Args>(args)...);
            __context::set(std::type_index(typeid(T)), ctx);
        }

        template<typename T>
        inline void finalize() {
            delete ptr<T>();
        }

        template<typename T>
        class bind_new {
        public:
            using self = bind_new;

            template<typename... Args>
            explicit bind_new(Args &&...args)
                    : m_ctx(std::type_index(typeid(T)), new T(std::forward<Args>(args)...)) {
                m_object = m_ctx.ctx();
            }

            ~bind_new() {
                delete m_object;
            }

            bind_new(const self &) = delete;

            self &operator=(const self &) = delete;

        private:
            __context m_ctx;
            T *m_object;
        };
    }
}


#endif //TENSORSTACK_UTILS_CTXMGR_H
