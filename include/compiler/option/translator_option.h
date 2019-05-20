#ifndef TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H
#define TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H

#include "module/graph.h"
#include "utils/static.h"

#include <set>

namespace ts {

    class TranslatorOption {
    public:
        using self = TranslatorOption;

        /**
        * translate node on device
        * @param device computing device
        * @param node checking node
        * @param translated_node get translated_node if return true
        * @param operators supported operator's name
        * @param output_flag mark current node is output or not.
        * @return translated return true, or false
        */
        virtual bool translate(const ComputingDevice &device, 
                               Node node, 
                               Node &zipped_node, 
                               std::set<const std::string> operators,
                               bool output_flag = false) const = 0;
    };

    const std::map<const TranslatorOption*, std::set<const std::string>> &GetFullTranslateOptions();

    void RegisterTranslateOption(const TranslatorOption *option, std::initializer_list<const std::string>& op_list);
    //void RegisterTranslateOption(const TranslatorOption *option, std::vector<const std::string>& op_list) {
}

#define TS_REGISTER_TRANSLATOR_OPTION(option,op_list) \
namespace { \
    static option ts_serial_name(_ts_translator_option); \
    TS_STATIC_ACTION(ts::RegisterTranslateOption, &(ts_serial_name(_ts_translator_option)),op_list); \
}

//#define TS_REGISTER_OPERATOR_ON_OPTION(option,op) \
//namespace { \
//    TS_STATIC_ACTION(ts::RegisterTranslateOption, &(ts_serial_name(_ts_translator_option))); \
//}

#endif //TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H
