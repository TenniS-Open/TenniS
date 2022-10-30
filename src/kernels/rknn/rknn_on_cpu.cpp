//
// Created by kier on 20-6-5.
//

#include "rknn2.h"

#include "global/operator_factory.h"
#include "global/memory_device.h"

using namespace ts;
using namespace rknn;

TS_REGISTER_OPERATOR(Rknn2, "cpu", "rknn2")

TS_STATIC_ACTION(ts::ComputingMemory::Register, "rknn", ts::CPU)