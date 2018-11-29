//
// Created by kier on 2018/11/11.
//

#ifndef TENSORSTACK_UTILS_CTXMGR_LITE_SUPPORT_H
#define TENSORSTACK_UTILS_CTXMGR_LITE_SUPPORT_H

#include "ctxmgr_lite.h"

namespace ts {

    template<typename T>
    TS_LITE_THREAD_LOCAL
    typename __thread_local_lite_context<T>::context
            __thread_local_lite_context<T>::m_ctx = nullptr;

    template<typename T>
    typename __thread_local_lite_context<T>::context
    __thread_local_lite_context<T>::swap(typename __thread_local_lite_context<T>::context ctx) {
        auto pre_ctx = m_ctx;
        m_ctx = ctx;
        return pre_ctx;
    }

    template<typename T>
    void __thread_local_lite_context<T>::set(typename __thread_local_lite_context<T>::context ctx) {
        m_ctx = ctx;
    }

    template<typename T>
    typename __thread_local_lite_context<T>::context const __thread_local_lite_context<T>::get() {
        if (m_ctx == nullptr) throw NoLiteContextException();
        return m_ctx;
    }

    template<typename T>
    typename __thread_local_lite_context<T>::context const __thread_local_lite_context<T>::try_get() {
        return m_ctx;
    }

    template<typename T>
    __lite_context<T>::__lite_context(typename __lite_context<T>::context ctx) {
        this->m_now_ctx = ctx;
        this->m_pre_ctx = __thread_local_lite_context<T>::swap(ctx);
    }

    template<typename T>
    __lite_context<T>::~__lite_context() {
        __thread_local_lite_context<T>::set(this->m_pre_ctx);
    }

    template<typename T>
    void __lite_context<T>::set(typename __lite_context<T>::context ctx) {
        __thread_local_lite_context<T>::set(ctx);
    }

    template<typename T>
    typename __lite_context<T>::context __lite_context<T>::get() {
        return __thread_local_lite_context<T>::get();
    }

    template<typename T>
    typename __lite_context<T>::context __lite_context<T>::try_get() {
        return __thread_local_lite_context<T>::try_get();
    }

    template<typename T>
    typename __lite_context<T>::context __lite_context<T>::ctx() {
        return m_now_ctx;
    }

    template<typename T>
    typename __lite_context<T>::context const __lite_context<T>::ctx() const {
        return m_now_ctx;
    }
}

#define TS_LITE_CONTEXT(T) \
    template class ts::__thread_local_lite_context<T>; \
    template class ts::__lite_context<T>;

#endif  // TENSORSTACK_UTILS_CTXMGR_LITE_SUPPORT_H