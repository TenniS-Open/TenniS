//
// Created by sen on 2021/12/21.
//

#ifndef TENNIS_THREADPOOL_H
#define TENNIS_THREADPOOL_H

#include "kernels/xnnpack/xnnpack.h"
#include "utils/ctxmgr_lite.h"

namespace ts {
    namespace xnn {
        class XnnThreadPool : public SetupContext<XnnThreadPool> {
        public:
            static XnnThreadPool &set_thread_number(int size) {
                static XnnThreadPool pool(size);
                return pool;
            }

            pthreadpool_t get_thread_pool() {
                return m_threadpool;
            }

            pthreadpool_t reset_thread_pool(int new_size) {
                if (m_threadpool) {
                    pthreadpool_destroy(m_threadpool);
                    m_threadpool = pthreadpool_create(new_size);
                }
                return m_threadpool;
            }

            XnnThreadPool(XnnThreadPool const& threadpool) = delete;
            XnnThreadPool &operator=(XnnThreadPool const& threadpool) = delete;

        private:
            explicit XnnThreadPool(int size);
            ~XnnThreadPool();
            pthreadpool_t m_threadpool = nullptr;
        };

    }
}

#endif //TENNIS_THREADPOOL_H
