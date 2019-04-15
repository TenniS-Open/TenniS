//
// Created by kier on 2018/12/21.
//

#include <runtime/runtime.h>

#include "runtime/runtime.h"

#include "utils/ctxmgr_lite_support.h"

#include <algorithm>

namespace ts {
    RuntimeContext::RuntimeContext():
        m_computing_thread_number(1) {
        this->m_thread_pool = std::make_shared<ThreadPool>(0);
    }

    void RuntimeContext::set_computing_thread_number(int computing_thread_number) {
        this->m_computing_thread_number = std::max(computing_thread_number, 1);
        this->m_thread_pool = std::make_shared<ThreadPool>(this->m_computing_thread_number);
    }

    int RuntimeContext::get_computing_thread_number() const {
        return m_computing_thread_number;
    }

    RuntimeContext::self RuntimeContext::clone() const {
        self doly;
        doly.m_computing_thread_number = this->m_computing_thread_number;
        if (m_thread_pool) {
            doly.m_thread_pool = std::make_shared<ThreadPool>(this->m_thread_pool->size());
        }
        if (this->m_dynamic) {
            doly.m_dynamic = this->m_dynamic->clone();
        }
        if (this->m_flow) {
            doly.m_flow = this->m_flow->clone();
        }
        return std::move(doly);
    }

    RuntimeContext::RuntimeContext(RuntimeContext::self &&other) {
        this->operator=(std::move(other));
    }

    RuntimeContext::self &RuntimeContext::operator=(RuntimeContext::self &&other) {
        std::swap(this->m_computing_thread_number, other.m_computing_thread_number);
        std::swap(this->m_thread_pool, other.m_thread_pool);
        std::swap(this->m_dynamic, other.m_dynamic);
        std::swap(this->m_flow, other.m_flow);
        return *this;
    }

    ThreadPool &RuntimeContext::thread_pool() {
        return *this->m_thread_pool;
    }

    void RuntimeContext::bind_flow(SyncMemoryController::shared flow) {
        m_flow = std::move(flow);
    }

    void RuntimeContext::bind_dynamic(SyncMemoryController::shared dynamic) {
        m_dynamic = std::move(dynamic);
    }

    SyncMemoryController::shared RuntimeContext::flow() const {
        return m_flow;
    }

    SyncMemoryController::shared RuntimeContext::dynamic() const {
        return m_dynamic;
    }

    SyncMemoryController::shared RuntimeContext::FlowMemory() {
        auto runtime = ctx::get<RuntimeContext>();
        if (!runtime) return nullptr;
        return runtime->flow();
    }

    SyncMemoryController::shared RuntimeContext::DynamicMemory() {
        auto runtime = ctx::get<RuntimeContext>();
        if (!runtime) return nullptr;
        return runtime->dynamic();}
}

TS_LITE_CONTEXT(ts::RuntimeContext)
