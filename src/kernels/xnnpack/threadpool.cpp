//
// Created by sen on 2021/12/21.
//

#include "kernels/xnnpack/threadpool.h"
#include "utils/ctxmgr_lite_support.h"

ts::xnn::XnnThreadPool::XnnThreadPool(int size) {
    m_threadpool = pthreadpool_create(size);
}

ts::xnn::XnnThreadPool::~XnnThreadPool() {
    pthreadpool_destroy(m_threadpool);
}

TS_LITE_CONTEXT(ts::xnn::XnnThreadPool)
