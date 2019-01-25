//
// Created by kier on 2019/1/25.
//

#include <runtime/image_filter.h>

#include "runtime/image_filter.h"

#include "utils/ctxmgr_lite.h"
#include "module/menu.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {

    ImageFilter::ImageFilter(const ComputingDevice &device) {
        this->m_computing_device = device;
        this->clear();
    }

    void ImageFilter::clear() {
        this->m_workbench.reset();
        this->m_graph = std::make_shared<Graph>();
        ctx::bind<Graph> _bind_graph(this->m_graph.get());
        bubble::param(serial_name(), {-1, -1, -1});    // add input param to graph
        this->m_compiled = false;
    }

    void ImageFilter::compile() {
        if (this->m_graph->nodes().size() > 1) {
            Module::shared module = std::make_shared<Module>();
            module->load(*m_graph);
            this->m_workbench = Workbench::Load(module, m_computing_device);
        }
        this->m_compiled = true;
    }

    Tensor ImageFilter::run(const Tensor &image) {
        if (!this->m_compiled) this->compile();
        if (!this->m_workbench) return image;

        this->m_workbench->input(0, image);
        this->m_workbench->run();
        return this->m_workbench->output(0);
    }

    std::string ImageFilter::serial_name() const {
        return "_" + std::to_string(this->m_graph->nodes().size());
    }

    void ImageFilter::to_float() {
        ctx::bind<Graph> _bind_graph(this->m_graph.get());
        auto top = this->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::to_float(), {top});
        (void)(node);
        this->m_compiled = false;
    }

    void ImageFilter::scale(float f) {
        ctx::bind<Graph> _bind_graph(this->m_graph.get());
        auto lhs = this->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), tensor::build(FLOAT32, f));
        auto node = bubble::op(serial_name(), name::layer::mul(), {lhs, rhs});
        (void)(node);
        this->m_compiled = false;
    }

    void ImageFilter::sub_mean(const std::vector<float> &mean) {
        auto mean_tensor = tensor::build(FLOAT32, {1, 1, int(mean.size())}, mean);
        ctx::bind<Graph> _bind_graph(this->m_graph.get());
        auto lhs = this->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), mean_tensor);
        auto node = bubble::op(serial_name(), name::layer::sub(), {lhs, rhs});
        (void)(node);
        this->m_compiled = false;

    }

    void ImageFilter::resize(int width, int height) {
        auto size_tensor = tensor::build(INT32, {height, width, -1});
        ctx::bind<Graph> _bind_graph(this->m_graph.get());
        auto x = this->m_graph->nodes().back();
        auto size = bubble::data(serial_name(), size_tensor);;
        auto node = bubble::op(serial_name(), name::layer::resize2d(), {x, size});
        // node->set(name::type, tensor::from(0));  // set resize method
        (void)(node);
        this->m_compiled = false;
    }

    void ImageFilter::channel_swap(const std::vector<int> &shuffle) {
        auto shuffle_tensor = tensor::build(INT32, shuffle);
        auto dim_tensor = tensor::build(INT32, {2, });
        auto x = this->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::dimshuffle(), {x});
        node->set(name::dim, dim_tensor);
        node->set(name::shuffle, shuffle_tensor);
        this->m_compiled = false;
    }

    void ImageFilter::to_chw() {
        auto permute_tensor = tensor::build(INT32, {2, 0, 1});
        auto x = this->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::transpose(), {x});
        node->set(name::permute, permute_tensor);
        this->m_compiled = false;
    }

    void ImageFilter::center_crop(int width, int height) {
        ctx::bind<Graph> _bind_graph(this->m_graph.get());
        auto x = this->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::hwc_center_crop2d(), {x});
        node->set(name::size, tensor::build(INT32, {width, height}));  // set resize method
        (void)(node);
        this->m_compiled = false;

    }

}
