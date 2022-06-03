//
// Created by sen on 2021/12/21.
//

#include "kernels/xnnpack/threadpool.h"
#include "utils/ctxmgr_lite_support.h"
#include "utils/assert.h"

#include <mutex>

namespace {
	std::mutex mutex;
	int count = 0;
	void increase() {
		std::lock_guard<decltype(mutex)> _lock(mutex);
		if (count == 0) {
			xnn_status status = xnn_initialize(nullptr);
			TS_CHECK(status == xnn_status_success);
		}
		++count;
	}
	void decrease() {
		std::lock_guard<decltype(mutex)> _lock(mutex);
		--count;
		if (count == 0) {
			xnn_status status = xnn_deinitialize();
			TS_CHECK(status == xnn_status_success);
		}
	}
}

ts::xnn::XnnThreadPool::XnnThreadPool(int size) {
	increase();
    m_threadpool = pthreadpool_create(size);
}

ts::xnn::XnnThreadPool::~XnnThreadPool() {
    TS_CHECK(m_threadpool != nullptr);
    pthreadpool_destroy(m_threadpool);
    m_threadpool = nullptr;
	decrease();
}

TS_LITE_CONTEXT(ts::xnn::XnnThreadPool)
