#ifndef TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H
#define TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H

#include "module/graph.h"
#include "utils/static.h"
#include "module/module.h"

namespace ts {

    class TranslatorOption {
    public:
        using self = TranslatorOption;

        virtual ~TranslatorOption() = default;

        /**
         * translate node on device
         * @param device computing device
         * @param node wait to translate
         * @param translated node
         * @param output_flag true when current node is output,otherwise false
         * @return translated return true, or false
         */

        virtual bool translate(const ComputingDevice &device,
            const Node node,
            Node &translated_node,
            bool output_flag = false) const = 0;
    };

    const std::vector<const TranslatorOption*> &GetFullTranslateOptions();

    void RegisterTranslateOption(const TranslatorOption *option);
}

#define TS_REGISTER_TRANSLATOR_OPTION(option) \
namespace { \
    static option ts_serial_name(_ts_translator_option); \
    TS_STATIC_ACTION(ts::RegisterTranslateOption, &(ts_serial_name(_ts_translator_option))); \
}

#endif //TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H
