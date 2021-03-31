//
// Created by Kier on 2021/3/29.
//

#include <module/io/sstream.h>
#include "compiler/fence/splitter.h"
#include "module/menu.h"

using namespace ts;

class Rule : public SplitRule {
public:
    bool is_route(const Node &node) const final {
        return m_route.find(node->op()) != m_route.end();
    }

    bool is_support(const Node &node) const final {
        return m_support.find(node->op()) != m_support.end();
    }

private:
    std::unordered_set<std::string> m_support = {"add"};
    std::unordered_set<std::string> m_route;
};

int main() {
    Graph _g;
    _g.setup_context();

    // a b
    // |/
    // c
    // |\
    // d e
    // |/
    // f
    // |\
    // g h

    auto a = bubble::param("a");
    auto b = bubble::param("b");
    auto c = bubble::op("c", "add", {a, b});
    auto d = bubble::op("d", "add", {c});
    auto e = bubble::op("e", "add", {c});
    auto f = bubble::op("f", "add", {d, e});
    auto g = bubble::op("g", "add", {f});
    auto h = bubble::op("h", "add", {f});

    // a b
    // |/|
    // c |
    // |\|
    // d e
    // | |
    // f g

//    auto a = bubble::param("a");
//    auto b = bubble::param("b");
//    auto c = bubble::op("c", "add", {a, b});
//    auto d = bubble::op("d", "add", {c});
//    auto e = bubble::op("e", "add", {c, b});
//    auto f = bubble::op("f", "add", {d});
//    auto g = bubble::op("g", "add", {e});

    std::vector<Node> bottoms = {a, b};
    std::vector<Node> tops = {g, h};

    Rule rule;
    Splitter splitter(&rule);
    splitter.single_input = false;

    MainGraph main = splitter.split(tops, bottoms);

    std::cout << "================ Main ===============" << std::endl;
    plot_graph(std::cout, main.outputs());
    std::cout << "================= Sub ===============" << std::endl;
    for (size_t i = 0; i < main.sub_count(); ++i) {
        std::cout << "================ Sub " << i << " ===============" << std::endl;
        auto &node = main.sub_node(i);
        auto &graph = main.sub_graph(i);
        plot_graph(std::cout, graph.outputs());

        auto submodule = std::make_shared<Module>();
        submodule->load(graph.graph(), graph.outputs());
        submodule->sort_inputs(graph.inputs());

        StringStreamWriter ssw;
        Module::Save(ssw, submodule);
        auto binary = ssw.str();

        node->name("submodule");
        node->set("module", tensor::build(UINT8,
                                          Shape({int(binary.size()),}),
                                          reinterpret_cast<const uint8_t *>(binary.data())));
    }

    main.module();

    return 0;
}

