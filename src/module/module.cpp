//
// Created by kier on 2018/7/18.
//

#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <utility>
#include <climits>
#include <algorithm>

#include <module/module.h>


#include "module/module.h"
#include "utils/box.h"
#include "core/tensor_builder.h"
#include "module/io/fstream.h"
#include "module/menu.h"
#include "module/header.h"

namespace ts {
    const char *const Bubble::Parameter = "<param>";
    const char *const Bubble::Const = "<const>";
    const char *const Bubble::Variable = "<var>";
    static const std::unordered_set<std::string> EndPoints = {Bubble::Parameter, Bubble::Const, Bubble::Variable};

    std::string Bubble::RetentionParam::name = "#name";
    std::string Bubble::RetentionParam::op = "#op";
    std::string Bubble::RetentionParam::output_count = "#output_count";
    std::string Bubble::RetentionParam::shape = "#shape";

    bool Bubble::IsEndPoint(const std::string &op) {
        return EndPoints.find(op) != EndPoints.end();
    }

    Bubble::Bubble(self &&other) TS_NOEXCEPT {
        this->operator=(std::forward<self>(other));
    }

    Bubble::self &Bubble::operator=(self &&other) TS_NOEXCEPT {
        this->m_op = std::move(other.m_op);
        this->m_name = std::move(other.m_name);
        this->m_output_count = other.m_output_count;
        this->m_params = std::move(other.m_params);
        return *this;
    }

    Bubble::Bubble(const std::string &op)
            : m_op(op) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name)
            : m_op(op), m_name(name) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, int output_count)
            : m_op(op), m_output_count(output_count) {
        TS_AUTO_CHECK(output_count >= 1);
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name, int output_count)
            : m_op(op), m_name(name), m_output_count(output_count) {
        TS_AUTO_CHECK(output_count >= 1);
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const Shape &shape)
            : m_op(op), m_shape(shape) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name, const Shape &shape)
            : m_op(op), m_name(name), m_shape(shape) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, int output_count, const Shape &shape)
            : m_op(op), m_output_count(output_count), m_shape(shape) {
        TS_AUTO_CHECK(output_count >= 1);
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name, int output_count, const Shape &shape)
            : m_op(op), m_name(name), m_output_count(output_count), m_shape(shape) {
        TS_AUTO_CHECK(output_count >= 1);
        update_retention_params();
    }

    bool Bubble::has(const std::string &param) const {
        return this->m_params.find(param) != this->m_params.end();
    }

    void Bubble::set(const std::string &param, const Tensor &value) {
        this->m_params.insert(std::make_pair(param, value));
    }

    Tensor &Bubble::get(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        return param_it->second;
    }

    const Tensor &Bubble::get(const std::string &param) const {
        return const_cast<self *>(this)->get(param);
    }

    void Bubble::clear(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        this->m_params.erase(param_it);
    }

    void Bubble::clear_params() {
        std::vector<std::pair<std::string, Tensor>> retention_params;
        for (auto &param_tenosr_pair : m_params) {
            auto &param = param_tenosr_pair.first;
            bool is_retention_param = !param.empty() && param[0] == retention_param_sign;
            if (is_retention_param) {
                retention_params.emplace_back(param_tenosr_pair);
            }
        }
        m_params.clear();
        m_params.insert(retention_params.begin(), retention_params.end());
    }

    std::string Bubble::fuzzy_param_name(const std::string &name) {
        if (m_params.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &param_tensor_pair : m_params) {
            auto &target_name = param_tensor_pair.first;
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    void Bubble::update_retention_params() {
        TS_AUTO_CHECK(retention_param_sign == '#');
        set(RetentionParam::op, tensor::from(m_op));
        set(RetentionParam::name, tensor::from(m_name));
        set(RetentionParam::output_count, tensor::from(m_output_count));
        if (!m_shape.empty()) {
            set(RetentionParam::shape, tensor::from(m_shape));
        }
    }

    static size_t write_param(StreamWriter &stream, const std::pair<std::string, Tensor> &param) {
        auto &name = param.first;
        auto &value = param.second;
        size_t writen_size = 0;
        // 1 write param's name
        // 1.1 write name length
        writen_size += binio::write<uint32_t>(stream, uint32_t(name.size()));
        // 1.2 write name string
        writen_size += binio::write<char>(stream, name.data(), name.size());
        // 2. write param's value
        writen_size += value.serialize(stream);
        return writen_size;
    }

    static size_t read_param(StreamReader &stream, std::pair<std::string, Tensor> &param) {
        auto &name = param.first;
        auto &value = param.second;
        size_t read_size = 0;
        uint32_t size_buffer;
        // 1. read param's name
        // 1.1 read name length
        read_size += binio::read<uint32_t>(stream, size_buffer);
        // 1.2 read name
        std::vector<char> string_buffer(size_buffer);
        read_size += binio::read<char>(stream, string_buffer.data(), size_buffer);
        name = std::string(string_buffer.begin(), string_buffer.end());
        // 2. read param's value
        read_size += value.externalize(stream);
        return read_size;
    }

    size_t Bubble::serialize(StreamWriter &stream) const {
        size_t writen_size = 0;
        writen_size += binio::write<uint32_t>(stream, uint32_t(m_params.size()));
        for (auto &param : m_params) {
            writen_size += write_param(stream, param);
        }
        return writen_size;
    }

    size_t Bubble::externalize(StreamReader &stream) {
        m_params.clear();
        size_t read_size = 0;
        uint32_t size_buffer;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        std::pair<std::string, Tensor> param;
        for (uint32_t i = 0; i < size_buffer; ++i) {
            read_size += read_param(stream, param);
            m_params.insert(param);
        }
        m_op = tensor::to_string(m_params[RetentionParam::op]);
        m_name = tensor::to_string(m_params[RetentionParam::name]);
        m_output_count = tensor::to_int(m_params[RetentionParam::output_count]);
        {
            auto it = m_params.find(RetentionParam::shape);
            if (it != m_params.end()) {
                auto tshape = tensor::cast(INT32, it->second);
                m_shape.resize(tshape.count());
                for (size_t i = 0; i < m_shape.size(); ++i) {
                    m_shape[i] = tshape.data<int32_t>(i);
                }
            }
        }
        return read_size;
    }

    void Module::load(Graph g) {
        auto nodes = g.nodes();
        std::vector<Node> outputs;
        for (auto &node : nodes) {
            if (node.outputs().empty()) {
                outputs.push_back(node);
            }
        }
        load(g, outputs);
    }

    void Module::load(Graph g, const std::vector<Node> &outputs) {
        auto inputs = graph_walker(g, outputs);

        m_inputs.insert(m_inputs.end(), inputs.begin(), inputs.end());
        m_outputs.insert(m_outputs.end(), outputs.begin(), outputs.end());
        m_graphs.push_back(g);
    }

    void Module::load(Graph g, const std::vector<std::string> &outputs) {
        std::unordered_map<std::string, Node> map_name_found_node;
        size_t found_count = 0;
        Graph local_graph;
        Node empty_node = local_graph.make<Bubble>("_empty");
        for (auto &output_name : outputs) {
            map_name_found_node.insert(std::make_pair(output_name, empty_node));
        }
        auto nodes = g.nodes();
        for (auto &node : nodes) {
            auto &bubble = node.ref<Bubble>();
            auto name_it = map_name_found_node.find(bubble.name());
            if (name_it == map_name_found_node.end()) {
                continue;
            }
            if (name_it->second != empty_node) {
                TS_LOG_ERROR << "Found duplicate Node " << node.str() << ", with Node " << name_it->second.str()
                             << eject;
            }
            ++found_count;
            name_it->second = node;
        }
        if (found_count != map_name_found_node.size()) {
            std::ostringstream oss;
            oss << "Can not found those names in graph: ";
            size_t not_found_name_count = 0;
            for (auto &name_found_node_pair : map_name_found_node) {
                if (name_found_node_pair.second != empty_node) continue;
                if (not_found_name_count) oss << ", ";
                oss << name_found_node_pair.first;
                ++not_found_name_count;
            }
            TS_LOG_ERROR(oss.str()) << eject;
        }
        std::vector<Node> found_nodes;
        found_nodes.reserve(outputs.size());
        for (auto &output_name : outputs) {
            Node &found_node = map_name_found_node.at(output_name);
            found_nodes.emplace_back(found_node);
        }
        this->load(g, found_nodes);
    }

    std::vector<Node> Module::graph_walker(Graph g, const std::vector<Node> &outputs) {
        // calculate inputs and outputs
        auto nodes = g.nodes();
        std::unordered_set<Node> nodes_set;
        for (auto &node : nodes) {
            nodes_set.insert(node);
        }
        // valid graph, get inputs
        std::vector<Node> inputs;
        std::queue<Node> walker;
        std::unordered_set<Node> walked;
        for (auto &node : outputs) walker.push(node);
        while (!walker.empty()) {
            auto node = walker.front();
            walker.pop();
            if (walked.find(node) != walked.end()) continue;
            walked.insert(node);
            if (nodes_set.find(node) == nodes_set.end()) {
                TS_LOG_ERROR << "Found unlinked node in graph: " << node.str() << eject;
            }
            auto &op = node.ref<Bubble>();
            if (op.op() == Bubble::Parameter) {
                inputs.push_back(node);
                continue;
            }
            if (op.op() == Bubble::Const) {
                continue;
            }
            if (node.inputs().empty()) {
                TS_LOG_ERROR << "Found not computable node: " << node.str() << eject;
            }
            for (auto &input : node.inputs()) {
                walker.push(input);
            }
        }
        // remove duplicate inputs
        std::unordered_set<Node> input_nodes_set;
        for (auto &input : inputs) input_nodes_set.insert(input);

        // return non-duplicate nodes
        return std::vector<Node>(input_nodes_set.begin(), input_nodes_set.end());
    }

    void Module::clear() {
        m_inputs.clear();
        m_outputs.clear();
        m_graphs.clear();
    }

    void Module::sort_inputs(const std::vector<Node> &inputs) {
        std::unordered_set<Node> sorted_input_set(inputs.begin(), inputs.end());
        for (auto &had_input : m_inputs) {
            if (sorted_input_set.find(had_input) == sorted_input_set.end()) {
                TS_LOG_ERROR << "The sorted inputs must content " << had_input.str() << eject;
            }
        }
        m_inputs = inputs;
    }

    void Module::sort_inputs(const std::vector<std::string> &input_names) {
        std::unordered_map<std::string, Node> map_name_input_node;
        // map nodes
        for (auto &input : m_inputs) {
            auto &bubble = input.ref<Bubble>();
            if (map_name_input_node.find(bubble.name()) != map_name_input_node.end()) {
                auto it = map_name_input_node.find(bubble.name());
                TS_LOG_ERROR << "Can not sort inputs with duplicate names: " << input.str() << " and "
                             << it->second.str() << eject;
            }
            map_name_input_node.insert(std::make_pair(bubble.name(), input));
        }
        // check inputs node
        std::unordered_set<Node> used_inputs;   // make sure all inputs must be used after used
        std::vector<Node> sorted_inputs;
        for (auto &input_name : input_names) {
            auto name_node_it = map_name_input_node.find(input_name);
            if (name_node_it == map_name_input_node.end()) {
                TS_LOG_ERROR << "Can not recognize name " << input_name << eject;
            }
            auto &node = name_node_it->second;
            sorted_inputs.emplace_back(node);
            used_inputs.insert(node);
        }
        if (used_inputs.size() < map_name_input_node.size()) {
            std::ostringstream oss;
            oss << "All inputs must be used after sorted, missing: ";
            size_t missing_count = 0;
            for (auto &name_node_pair : map_name_input_node) {
                auto &node = name_node_pair.second;
                if (used_inputs.find(node) != used_inputs.end()) continue;
                if (missing_count) oss << ", ";
                oss << name_node_pair.first;
                ++missing_count;
            }
            TS_LOG_ERROR(oss.str()) << eject;
        }
        m_inputs = sorted_inputs;
    }

    void Module::sort_inputs(const std::initializer_list<std::string> &input_names) {
        this->sort_inputs(std::vector<std::string>(input_names.begin(), input_names.end()));
    }

    template <typename K, typename V>
    using map = std::unordered_map<K, V>;
    template <typename K>
    using set = std::unordered_set<K>;

    std::vector<std::pair<Node, int>> Module::list_reference_nodes(const std::vector<Node> &nodes) {
        map<Node, int> map_node_depth;
        std::deque<Node> node_walker; // top_down

        for (auto &node : nodes) {
            auto it = map_node_depth.find(node);
            if (it != map_node_depth.end()) continue;
            node_walker.push_back(node);
            map_node_depth.insert(std::make_pair(node, 1));
        }

        while (!node_walker.empty()) {
            auto node = node_walker.front();
            node_walker.pop_front();
            auto depth = map_node_depth[node];
            for (auto &input : node.inputs()) {
                auto input_depth_pair = map_node_depth.find(input);
                if (input_depth_pair == map_node_depth.end()) {
                    map_node_depth.insert(std::make_pair(input, depth + 1));
                    node_walker.push_back(input);
                } else {
                    auto this_input_depth = depth + 1;
                    if (input_depth_pair->second < this_input_depth) {
                        input_depth_pair->second = this_input_depth;
                        node_walker.push_back(input);   // re-walk nodes
                    }
                }
            }
        }

        std::vector<std::pair<Node, int>> computation_schedule(map_node_depth.begin(), map_node_depth.end());
        std::sort(computation_schedule.begin(), computation_schedule.end(),
                  [](const std::pair<Node, int> &lhs, const std::pair<Node, int> &rhs){
                      return lhs.second > rhs.second;
                  });

        return std::move(computation_schedule);
    }

    void Module::Save(const std::string &filename, Module::shared module, Module::SerializationFormat format) {
        TS_AUTO_CHECK(format == BINARY);
        // get nodes ready to read
        FileStreamWriter stream(filename);
        auto valued_nodes = list_reference_nodes(module->outputs());
        std::vector<Node> nodes;
        std::unordered_map<Node, size_t> map_node_index;
        size_t index = 0;
        for (auto &valued_node : valued_nodes) {
            auto &node = valued_node.first;
            map_node_index.insert(std::make_pair(node, index++));
            nodes.emplace_back(node);
        }
        for (auto &input : module->inputs()) {
            if (map_node_index.find(input) != map_node_index.end()) continue;
            auto &node = input;
            map_node_index.insert(std::make_pair(node, index++));
            nodes.emplace_back(node);
        }

        // 0. save header
        Header header;
        header.code = TS_MODULE_CODE_V1;
        header.serialize(stream);

        // 1. save inputs
        binio::write<uint32_t>(stream, uint32_t(module->inputs().size()));
        for (auto &node : module->inputs()) {
            binio::write<uint32_t>(stream, uint32_t(map_node_index[node]));
        }
        // 2. save outputs
        binio::write<uint32_t>(stream, uint32_t(module->outputs().size()));
        for (auto &node : module->outputs()) {
            binio::write<uint32_t>(stream, uint32_t(map_node_index[node]));
        }
        // 3. save graphs
        serialize_nodes(stream, nodes);
    }

    static size_t read_uint32_list(StreamReader &stream, std::vector<uint32_t> &list) {
        uint32_t size_buffer = 0;
        size_t read_size = 0;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        list.resize(size_buffer);
        for (auto &elem : list) {
            read_size += binio::read<uint32_t>(stream, elem);
        }
        return read_size;
    }

    Module::shared Module::Load(const std::string &filename, Module::SerializationFormat format) {
        TS_AUTO_CHECK(format == BINARY);
        FileStreamReader stream(filename);
        size_t read_size = 0;

        // 0. read header
        Header header;
        read_size += header.externalize(stream);
        TS_AUTO_CHECK(header.code == TS_MODULE_CODE_V1);

        // 1. read inputs
        // read node index
        std::vector<uint32_t> input_index;
        read_size += read_uint32_list(stream, input_index);
        // 2. read outputs
        std::vector<uint32_t> output_index;
        read_size += read_uint32_list(stream, output_index);
        // 3. read graph
        Graph g;
        read_size += externalize_graph(stream, g);
        const auto &nodes = g.nodes();  // TODO: Check if the read nodes is the given nodes
        // x.1 convert inputs and outputs
        std::vector<Node> inputs;
        for (auto index : input_index) inputs.emplace_back(nodes[index]);
        std::vector<Node> outputs;
        for (auto index : output_index) outputs.emplace_back(nodes[index]);
        Module::shared module = std::make_shared<Module>();
        module->load(g, outputs);
        module->sort_inputs(inputs);
        return module;
    }
}
