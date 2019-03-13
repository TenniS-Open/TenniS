//
// Created by kier on 2018/12/21.
//

#ifndef TENSORSTACK_RUNTIME_RUNTIME_H
#define TENSORSTACK_RUNTIME_RUNTIME_H

#include "utils/ctxmgr.h"
#include "inside/thread_pool.h"

namespace ts {

    class TS_DEBUG_API RuntimeContext {
    public:
        using self = RuntimeContext;

        RuntimeContext();

        RuntimeContext(const self &) = delete;
        self &operator=(const self &) = delete;

        RuntimeContext(self &&other);
        self &operator=(self &&other);

        int get_computing_thread_number() const;

        void set_computing_thread_number(int computing_thread_number);

        self clone() const;

        ThreadPool &thread_pool();

    private:
        /**
         * Computing threads number. Used in OpenMP
         */
        int m_computing_thread_number = 1;

        ThreadPool::shared m_thread_pool;
    };

}


#endif //TENSORSTACK_RUNTIME_RUNTIME_H
