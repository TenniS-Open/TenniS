#include "global/module_loader_factory.h"
#include "utils/static.h"

#include <map>

namespace ts {
    static std::map<std::string, ModuleLoader::function> &MapModuleLoader() {
        static std::map<std::string, ModuleLoader::function> map_module_loader;
        return map_module_loader;
    }

    ModuleLoader::function ModuleLoader::Query(const std::string &module) TS_NOEXCEPT {
        auto &map_module_loader = MapModuleLoader();
        auto module_loader = map_module_loader.find(module);
        if (module_loader != map_module_loader.end()) {
            return module_loader->second;
        }
        return ModuleLoader::function(nullptr);
    }

    void ModuleLoader::Register(const std::string &module, const function &loader) TS_NOEXCEPT {
        auto &map_module_loader = MapModuleLoader();
        map_module_loader[module] = loader;
    }

    std::set<std::string> ModuleLoader::AllKeys() TS_NOEXCEPT {
        auto &map_module_loader = MapModuleLoader();
        std::set<std::string> set_module;
        for (auto &module : map_module_loader) {
            set_module.insert(module.first);
        }
        return set_module;
    }

    void ModuleLoader::Clear() {
        auto &map_module_loader = MapModuleLoader();
        map_module_loader.clear();
    }
}
