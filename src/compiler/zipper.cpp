//
// Created by kier on 2018/11/1.
//

#include "compiler/zipper.h"
#include "compiler/option/zipper_option.h"

#include <unordered_set>

namespace ts {

    Zipper::Zipper(const ComputingDevice &device)
            : m_device(device) {
    }

    static Node zip_node(const Node &node,
                         std::unordered_map<Node, Node> &ready_map,
                         const ComputingDevice &device,
                         const std::vector<const ZipperOption *> &options) {
        {
            auto ready_it = ready_map.find(node);
            if (ready_it != ready_map.end()) {
                return ready_it->second;
            }
        }

        Node zipped_node = node;
        for (auto &option : options) {
            if (option->zip(device, node, zipped_node)) break;
        }

        ready_map.insert(std::make_pair(node, zipped_node));

        return zipped_node;
    }

    std::vector<Node> Zipper::zip(const std::vector<Node> &nodes) const {
        if (ctx::get<Graph>() == nullptr) {
            TS_LOG_ERROR << "context:<ts::Graph> needed, but not given." << eject;
        }

        // TODO: add more options
        std::vector<const ZipperOption *> options = GetFullOptions();

        std::vector<Node> zipped_nodes;
        std::unordered_map<Node, Node> ready_map;
        for (auto &node : nodes) {
            zipped_nodes.emplace_back(zip_node(node, ready_map, m_device, options));
        }
        return std::move(zipped_nodes);
    }
}
