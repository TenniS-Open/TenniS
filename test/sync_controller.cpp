//
// Created by kier on 2018/12/26.
//

#include "core/sync/sync_controller.h"
#include "core/controller.h"

int main() {
    auto controller = ts::HypeSyncMemoryController<ts::DynamicMemoryController>::Make(true);

    int i = 10;

    ts::MemoryDevice cpu0(ts::CPU, 0);
    ts::MemoryDevice cpu1(ts::CPU, 1);

    auto a = controller->alloc(cpu0, 4, true);

    TS_LOG_INFO << "Write i=" << i;
    a.sync(cpu0, ts::SyncMemory::WRITE).data<int>()[0] = i;

    TS_LOG_INFO << "Got i=" << a.sync(cpu1).data<int>()[0];
}

