//
// Created by yang on 2020/2/26.
//

#include "global/operator_factory.h"
#include "runtime/importor.h"
#include "runtime/switcher.h"
#include "utils/cpu_info.h"
#include "utils/api.h"

#include <vector>
#include <mutex>

#if TS_PLATFORM_OS_WINDOWS
const std::string tennis_avx_fma_dll = "tennis_haswell.dll";
const std::string tennis_avx_dll = "tennis_sandy_bridge.dll";
const std::string tennis_sse_dll = "tennis_pentium.dll";
#elif TS_PLATFORM_OS_LINUX
const std::string tennis_avx_fma_dll = "libtennis_haswell.so";
const std::string tennis_avx_dll = "libtennis_sandy_bridge.so";
const std::string tennis_sse_dll = "libtennis_pentium.so";
#elif TS_PLATFORM_OS_MAC
const std::string tennis_avx_fma_dll = "libtennis_haswell.dylib";
const std::string tennis_avx_dll = "libtennis_sandy_bridge.dylib";
const std::string tennis_sse_dll = "libtennis_pentium.dylib";
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

    class TS_DEBUG_API Switcher{
    public:
        using self = Switcher;
        using shared = std::shared_ptr<self>;

        Switcher(){
            m_importer = std::make_shared<Importor>();
        };
        ~Switcher(){
            free();
        };

        bool auto_switch(const ComputingDevice &device){
            //switch instruction
            std::unique_lock<std::mutex> _locker(m_switcher_mutex);
            if (m_is_loaded)
                return true;
            std::vector<CPUFeature> features = {AVX, FMA};
            if(check_cpu_features({AVX, FMA})){
                m_is_loaded = m_importer->load(tennis_avx_fma_dll);
            }
            else if(check_cpu_features({AVX})){
                m_is_loaded = m_importer->load(tennis_avx_dll);
            }
            else if(check_cpu_features({SSE, SSE2})){
                m_is_loaded = m_importer->load(tennis_sse_dll);
            }
            else{
                TS_LOG_ERROR <<
                             "Minimum support for SSE instruction,Otherwise you need to compile a version that does not support any instruction set" << eject;
            }

            m_pre_creator_map.reset(ts_plugin_get_creator_map(), ts_plugin_free_creator_map);

            get_creator_map creator_map_fuc =
                    (get_creator_map)m_importer->get_fuc_address("ts_plugin_get_creator_map");
            free_creator_map free_creator_map_fuc =
                    (free_creator_map)m_importer->get_fuc_address("ts_plugin_free_creator_map");
            m_creator_map.reset(creator_map_fuc(), free_creator_map_fuc);
            ts_plugin_flush_creator(m_creator_map.get());

            return true;
        }

        bool is_load_dll(){
            return m_is_loaded;
        }

        Importor::shared importor(){
            return m_importer;
        }

    private:
        void free(){
//            OperatorCreator::Clear();
            ts_plugin_flush_creator(m_pre_creator_map.get());
        };

    private:
        std::shared_ptr<Importor> m_importer;
        std::shared_ptr<ts_op_creator_map> m_pre_creator_map;
        std::shared_ptr<ts_op_creator_map> m_creator_map;

        bool m_is_loaded = false;
        std::mutex m_switcher_mutex;
    };

    static Switcher& get_switcher(){
        static Switcher switcher;
        return switcher;
    }

    void SwitchControll::auto_switch(const ComputingDevice &device){
        if (!m_is_loaded)
            return;
        m_is_loaded = get_switcher().auto_switch(device);
        bind_context(device);
    }

    bool SwitchControll::is_load_dll(){
        return m_is_loaded;
    }

    void SwitchControll::bind_context(const ComputingDevice &device){
        if(!is_load_dll()){
            TS_LOG_ERROR << "Dynamic library not loaded, please call auto_switch first" << eject;
        }

        auto& switcher = get_switcher();
        ts_initial_device_context initial_device_context_fuc =
                (ts_initial_device_context)switcher.importor()->get_fuc_address("ts_plugin_initial_device_context");
        free_device_context free_device_context_fuc =
                (free_device_context)switcher.importor()->get_fuc_address("ts_plugin_free_device_context");
        ts_Device ts_device;
        ts_device.id = device.id();
        ts_device.type = device.type().c_str();
        m_device_context.reset(initial_device_context_fuc(&ts_device), free_device_context_fuc);
    }
}

