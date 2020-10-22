#include <module/header.h>
#include <global/module_loader_factory.h>
#include "module/module.h"
#include "module/menu.h"
#include "utils/static.h"

using namespace ts;

Module::shared open_loader(StreamReaderV2 &stream, const void *buffer, int32_t buffer_size,
                           Module::SerializationFormat format) {
    size_t read_size = 0;

    // 0. read header
    Header header;
    read_size += header.externalize(stream);
    if (header.code != TS_MODULE_CODE_V1) {
        throw FormatMismatchException("Module format mismatch.");
    }

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

TS_STATIC_ACTION(ModuleLoader::Register, "open_loader", open_loader)
