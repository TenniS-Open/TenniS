//
// Created by kier on 2018/11/1.
//

#include "compiler/translater.h"
#include "compiler/option/translator_option.h"

#include "module/menu.h"

namespace ts {
    Translator::Translator(const ComputingDevice &device)
        : m_device(device){

    }

    static Node translate_node(const Node& node,
                               std::unordered_map<Node, Node> &ready_map,
                               const ComputingDevice &device,
                               std::map<const TranslatorOption*, std::set<const std::string>> &options,
                               bool output_flag) {

        //check ready map
        auto ready_it = ready_map.find(node);
        if (ready_it != ready_map.end()) {
            return ready_it->second;
        }

        auto translated_node = node;
        for ( auto option : options ){
            if (option.first->translate(device, node, translated_node, option.second, output_flag)) {
                break;
            }
        }

        bool translated = false;
        std::vector<Node> translated_inputs;
        auto input_nodes = node.inputs();
        for (auto &node : input_nodes) {
            auto translated_node = translate_node(node, ready_map, device, options, false);
            if (translated_node != node) {
                translated = true;
            }
            translated_inputs.emplace_back(translated_node);
        }

        if (translated) {
            translated_node = bubble::bubble(translated_node.bubble());
            Node::Link(translated_node, translated_inputs);
        }
    }

    Module Translator::translate(const Module &module) const {

        Module new_module;

        Graph temp_graph;
        ctx::bind<Graph> _bind_graph(temp_graph);

        auto options = GetFullTranslateOptions();

        std::vector<Node> traslated_nodes;
        std::unordered_map<Node, Node> ready_map;

        auto output_nodes = module.outputs();
        for ( auto & node: output_nodes)
        {
            auto translated_node = translate_node(node, ready_map, m_device, options, true);
            traslated_nodes.emplace_back(translated_node);
        }

        new_module.Load(temp_graph, traslated_nodes);
        return std::move(new_module);
    }
}