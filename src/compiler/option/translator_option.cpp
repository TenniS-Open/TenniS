#include "compiler/option/translator_option.h"

namespace ts {

    static std::vector<const TranslatorOption*> &GetStaticTranslateOptions() {
        static std::vector<const TranslatorOption*> options;
        return options;
    }

    const std::vector<const TranslatorOption*> &GetFullTranslateOptions() {
        return GetStaticTranslateOptions();
    }

    void RegisterTranslateOption(const TranslatorOption *option) {
        auto &options = GetStaticTranslateOptions();
        options.emplace_back(option);
    }
}

