//
// Created by kier on 2019/1/25.
//

#include <runtime/image_filter.h>

#include "runtime/image_filter.h"

#include "utils/ctxmgr_lite.h"
#include "module/menu.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/workbench.h"

#include <numeric>

namespace ts {

    class ImageFilter::Implement {
    public:
        ComputingDevice m_computing_device;
        Workbench::shared m_workbench;
        Graph::shared m_graph;
        bool m_compiled = false;
    };

    ImageFilter::ImageFilter(const ComputingDevice &device) {
        m_impl->m_computing_device = device;
        this->clear();
    }

    void ImageFilter::clear() {
        m_impl->m_workbench.reset();
        m_impl->m_graph = std::make_shared<Graph>();
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        bubble::param(serial_name(), {-1, -1, -1});    // add input param to graph
        m_impl->m_compiled = false;
    }

    void ImageFilter::compile() {
        if (m_impl->m_graph->nodes().size() > 1) {
            Module::shared module = std::make_shared<Module>();
            module->load(*m_impl->m_graph);
            m_impl->m_workbench = Workbench::Load(module, m_impl->m_computing_device);
        }
        m_impl->m_compiled = true;
    }

    class ShapeTransformer {
    public:
        Shape before(const Shape &shape) {
            m_base = shape;
            switch (shape.size()) {
                case 0:
                    TS_LOG_ERROR << "Can not transform empty shape." << eject;
                    break;
                case 1:
                    return {1, shape[0], 1, 1};
                case 2:
                    return {1, shape[0], shape[1], 1};
                case 3:
                    return {1, shape[0], shape[1], shape[2]};
                case 4:
                    return shape;
                default:
                    return {shape[0], shape[1], shape[2],
                            std::accumulate(shape.begin() + 3, shape.end(), 1, std::multiplies<int>())};
            }
            return Shape();
        };
        Shape after(const Shape &shape) {
            return shape;
        }

    private:
        std::vector<int> m_base;
    };

    Tensor ImageFilter::run(const Tensor &image) {
        if (!m_impl->m_compiled) this->compile();
        if (!m_impl->m_workbench) return image;

        Tensor nhwc_image = image;
        ShapeTransformer transformer;

        nhwc_image = nhwc_image.reshape(transformer.before(nhwc_image.sizes()));

        m_impl->m_workbench->input(0, nhwc_image);
        m_impl->m_workbench->run();
        auto output = m_impl->m_workbench->output(0);

        output = output.reshape(transformer.after(output.sizes()));

        return output;
    }

    std::string ImageFilter::serial_name() const {
        return "_" + std::to_string(m_impl->m_graph->nodes().size());
    }

    void ImageFilter::to_float() {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto top = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::to_float(), {top});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::scale(float f) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto lhs = m_impl->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), tensor::build(FLOAT32, f));
        auto node = bubble::op(serial_name(), name::layer::mul(), {lhs, rhs});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::sub_mean(const std::vector<float> &mean) {
        auto mean_tensor = tensor::build(FLOAT32, {1, 1, 1, int(mean.size())}, mean);
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto lhs = m_impl->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), mean_tensor);
        auto node = bubble::op(serial_name(), name::layer::sub(), {lhs, rhs});
        (void)(node);
        m_impl->m_compiled = false;

    }

    void ImageFilter::resize(int width, int height) {
        auto size_tensor = tensor::build(INT32, {-1, height, width, -1});
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto size = bubble::data(serial_name(), size_tensor);;
        auto node = bubble::op(serial_name(), name::layer::resize2d(), {x, size});
        // node->set(name::type, tensor::from(0));  // set resize method
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::channel_swap(const std::vector<int> &shuffle) {
        auto shuffle_tensor = tensor::build(INT32, shuffle);
        auto dim_tensor = tensor::build(INT32, {3, });
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::dimshuffle(), {x});
        node->set(name::dim, dim_tensor);
        node->set(name::shuffle, shuffle_tensor);
        m_impl->m_compiled = false;
    }

    void ImageFilter::to_chw() {
        auto permute_tensor = tensor::build(INT32, {0, 3, 1, 2});
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::transpose(), {x});
        node->set(name::permute, permute_tensor);
        m_impl->m_compiled = false;
    }

    void ImageFilter::center_crop(int width, int height) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::nhwc_center_crop2d(), {x});
        node->set(name::size, tensor::build(INT32, {width, height}));  // set resize method
        (void)(node);
        m_impl->m_compiled = false;

    }

    ImageFilter::shared ImageFilter::clone() const {
        ImageFilter::shared dolly(new ImageFilter(*this->m_impl));
        return dolly;
    }

    ImageFilter::ImageFilter(const ImageFilter::Implement &other) {
        m_impl->m_computing_device = other.m_computing_device;
        this->clear();
        m_impl->m_workbench = other.m_workbench->clone();
        m_impl->m_compiled = true;
    }

}
