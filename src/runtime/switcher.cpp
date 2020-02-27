//
// Created by yang on 2020/2/26.
//

#include "runtime/importor.h"
#include "runtime/switcher.h"
#include "utils/cpu_info.h"
#include "utils/api.h"

#include <vector>

#if TS_PLATFORM_OS_WINDOWS
const std::string tennis_avx_fma_dll = "tennis_Haswell.dll";
const std::string tennis_avx_dll = "tennis_SandyBridge.dll";
const std::string tennis_sse_dll = "tennis_Pentium.dll";
#elif TS_PLATFORM_OS_LINUX
const std::string tennis_avx_fma_dll = "tennis_Haswell.so";
const std::string tennis_avx_dll = "tennis_SandyBridge.so";
const std::string tennis_sse_dll = "tennis_Pentium.so";
#endif

namespace ts{

    typedef ts_op_creator_map* (*get_creator_map)();
    typedef void (*free_creator_map)(ts_op_creator_map*);

    typedef ts_device_context* (*ts_initial_device_context) (const ts_Device *device);
    typedef void (*free_device_context)(ts_device_context*);

    static inline bool check_cpu_features(const std::initializer_list<CPUFeature>& features){
        for (auto &fea : features) {
            auto flag = check_cpu_feature(fea);
            if (!flag) {
                return false;
            }
        }
        return true;
    }

    Switcher::Switcher() { 
        m_importer = std::make_shared<Importor>();
    }

    Switcher::~Switcher() {
        free();
    }

    void Switcher::auto_switch(const ComputingDevice &device){
        //switch instruction
        std::vector<CPUFeature> features = {AVX, FMA};
        if(check_cpu_features({AVX, FMA})){
            m_importer->load(tennis_avx_fma_dll);
        }
        else if(check_cpu_features({AVX})){
            m_importer->load(tennis_avx_dll);
        }
        else if(check_cpu_features({SSE, SSE2})){
            m_importer->load(tennis_sse_dll);
        }

        get_creator_map creator_map_fuc =
                (get_creator_map)m_importer->get_fuc_address("ts_get_creator_map");
        free_creator_map free_creator_map_fuc =
                (free_creator_map)m_importer->get_fuc_address("ts_free_creator_map");
        m_creator_map.reset(creator_map_fuc(), free_creator_map_fuc);
        ts_flush_creator(m_creator_map.get());

        //bind context
        ts_initial_device_context initial_device_context_fuc =
                (ts_initial_device_context)m_importer->get_fuc_address("ts_initial_device_context");
        free_device_context free_device_context_fuc =
                (free_device_context)m_importer->get_fuc_address("ts_free_device_context");
        ts_Device ts_device;
        ts_device.id = device.id();
        ts_device.type = device.type().c_str();
        m_device_context.reset(initial_device_context_fuc(&ts_device), free_device_context_fuc);
    }

    void Switcher::free(){

    }

}

