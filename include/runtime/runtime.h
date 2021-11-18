//
// Created by kier on 2018/12/21.
//

#ifndef TENSORSTACK_RUNTIME_RUNTIME_H
#define TENSORSTACK_RUNTIME_RUNTIME_H

#include "utils/ctxmgr.h"
#include "inside/thread_pool.h"

#include "core/sync/sync_controller.h"

#include "utils/ctxmgr_lite.h"
#ifdef TS_USE_XNNPACK
#include "kernels/xnnpack/xnnpack.h"
#endif

namespace ts {

    class TS_DEBUG_API RuntimeContext : public SetupContext<RuntimeContext> {
    public:
        using self = RuntimeContext;

        RuntimeContext();

        explicit RuntimeContext(const MemoryDevice &device);

        RuntimeContext(const self &) = delete;
        self &operator=(const self &) = delete;

        RuntimeContext(self &&other);
        self &operator=(self &&other);

        int get_computing_thread_number() const;

        void set_computing_thread_number(int computing_thread_number);

        self clone() const;

        ThreadPool &thread_pool();

         void bind_flow(SyncMemoryController::shared flow);

         void bind_dynamic(SyncMemoryController::shared dynamic);

        SyncMemoryController::shared flow() const;

        SyncMemoryController::shared dynamic() const;

        static SyncMemoryController::shared FlowMemory();

        static SyncMemoryController::shared DynamicMemory();
#ifdef TS_USE_XNNPACK
        pthreadpool_t get_xnn_threadpool();
#endif

    private:
        /**
         * Computing threads number. Used in OpenMP
         */
        int m_computing_thread_number = 1;

        ThreadPool::shared m_thread_pool;
#ifdef TS_USE_XNNPACK
        using XnnThreadPool = std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>;
        XnnThreadPool m_xnn_pthreadpool = XnnThreadPool(pthreadpool_create(m_computing_thread_number), pthreadpool_destroy);
#endif

        SyncMemoryController::shared m_flow;
        SyncMemoryController::shared m_dynamic;
    };
}


#endif //TENSORSTACK_RUNTIME_RUNTIME_H
