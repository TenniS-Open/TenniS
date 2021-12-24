//
// Created by sen on 2021/12/21.
//

#include "kernels/xnnpack/threadpool.h"
#include "utils/ctxmgr_lite_support.h"
#include "utils/assert.h"

ts::xnn::XnnThreadPool::XnnThreadPool(int size) {
    xnn_status status;
    status = xnn_initialize(nullptr);
    TS_CHECK(status == xnn_status_success);
    m_threadpool = pthreadpool_create(size);
}

ts::xnn::XnnThreadPool::~XnnThreadPool() {
    TS_CHECK(m_threadpool != nullptr);
    pthreadpool_destroy(m_threadpool);
    m_threadpool = nullptr;
    xnn_status status;
    status = xnn_deinitialize();
    TS_CHECK(status == xnn_status_success);
}

TS_LITE_CONTEXT(ts::xnn::XnnThreadPool)
