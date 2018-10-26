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


namespace std {
    template<>
    struct hash<std::type_info> {
        std::size_t operator()(const std::type_info &key) const {
            using std::size_t;
            using std::hash;

            return key.hash_code();
        }
    };
}

namespace ts {

    class NoContextException : public Exception {
    public:
        NoContextException()
                : NoContextException(std::this_thread::get_id()) {
        }

        explicit NoContextException(const std::thread::id &id)
                : Exception(build_message(id)), m_thread_id(id) {
        }

    private:
        std::string build_message(const std::thread::id &id) {
            std::ostringstream oss;
            oss << "Empty context in thread: " << id;
            return oss.str();
        }

        std::thread::id m_thread_id;
    };


    class __thread_context {
    public:
        using self = __thread_context;
        using stack = std::stack<void *>;

        void push(const std::thread::id &id, void *ctx) {
            auto &local_ctx = this->m_ctx[id];
            local_ctx.push(ctx);
        }

        void push(void *ctx) {
            auto id = std::this_thread::get_id();
            this->push(id, ctx);
        }

        const void *top(const std::thread::id &id) const {
            auto it = this->m_ctx.find(id);
            if (it == this->m_ctx.end()) throw NoContextException(id);
            auto &local_ctx = it->second;
            if (local_ctx.empty()) throw NoContextException();
            return local_ctx.top();
        }

        const void *top() const {
            auto id = std::this_thread::get_id();
            return this->top(id);
        }

        void *top(const std::thread::id &id) {
            return const_cast<void *>(const_cast<const self *>(this)->top(id));
        }

        void *top() {
            return const_cast<void *>(const_cast<const self *>(this)->top());
        }

        void pop(const std::thread::id &id) {
            auto it = this->m_ctx.find(id);
            if (it == this->m_ctx.end()) return;
            auto &local_ctx = it->second;
            if (!local_ctx.empty()) local_ctx.pop();
        }

        void pop() {
            auto id = std::this_thread::get_id();
            this->pop(id);
        }

        size_t size(const std::thread::id &id) const {
            auto it = this->m_ctx.find(id);
            if (it == this->m_ctx.end()) return 0;
            auto &local_ctx = it->second;
            return local_ctx.size();
        }

        size_t size() const {
            auto id = std::this_thread::get_id();
            return this->size(id);
        }

        void clear(const std::thread::id &id) {
            m_ctx.erase(id);
        }

    private:
        std::unordered_map<std::thread::id, stack> m_ctx;
    };


    extern std::unordered_map<std::type_index, __thread_context> __global_thread_context;

    class __context {
    public:
        using self = __context;

        explicit __context(const std::type_index &type, void *ctx) {
            auto &local_thread_context = __global_thread_context[type];
            this->m_id = std::this_thread::get_id();
            this->m_context = &local_thread_context;
            this->m_context->push(this->m_id, ctx);
        }

        ~__context() {
            this->m_context->pop(this->m_id);
            if (this->m_context->size(this->m_id) == 0) {
                this->m_context->clear(this->m_id);
            }
        }

        static void push(const std::type_index &type, void *ctx) {
            auto &local_thread_context = __global_thread_context[type];
            return local_thread_context.push(ctx);
        }

        static void pop(const std::type_index &type) {
            auto &local_thread_context = __global_thread_context[type];
            auto id = std::this_thread::get_id();
            local_thread_context.pop(id);
            if (local_thread_context.size(id) == 0) {
                local_thread_context.clear(id);
            }
        }

        static void *top(const std::type_index &type) {
            auto &local_thread_context = __global_thread_context[type];
            return local_thread_context.top();
        }

        __context(const self &) = delete;

        self &operator=(const self &) = delete;

    private:
        std::thread::id m_id;
        __thread_context *m_context = nullptr;
    };

    // TODO: 考虑线程安全问题，多线程初始化会发生什么事情呢~
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
        inline void push(T *ctx) {
            __context::push(std::type_index(typeid(T)), ctx);
        }

        template<typename T>
        inline void push(T &ctx_ref) {
            push<T>(&ctx_ref);
        }

        template<typename T>
        inline void pop() {
            __context::pop(std::type_index(typeid(T)));
        }

        template<typename T>
        inline T *get() {
            return reinterpret_cast<T *>(__context::top(std::type_index(typeid(T))));
        }

        template<typename T>
        inline T *ptr() { return get<T>(); }

        template<typename T>
        inline T &ref() { return *get<T>(); }
    }
}


#endif //TENSORSTACK_UTILS_CTXMGR_H
