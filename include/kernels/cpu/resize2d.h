#ifndef TS_KERNELS_RESIZE2D_H
#define TS_KERNELS_RESIZE2D_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>

namespace ts {


//support UINT8,UINT16,FLOAT32,FLOAT64, channel 1,2,3
class Resize2d : public ts::Operator {
public:


    template<typename T>
    static void ResizeImageLinear( const T *src_im, int src_width, int src_height, int channels,
                                     T *dst_im, int dst_width, int dst_height);

    template<typename T>
    static void ResizeImageCubic(const T *src_im, int src_width, int src_height, int channels,
                             T *dst_im, int dst_width, int dst_height);

public:
    enum RESIZETYPE{
        linear,
        cublic
    };

    using supper = ts::Operator;
    Resize2d();

    virtual void init();

    virtual int run(ts::Stack &stack);

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

private:
template<typename T>
    void resize(Tensor *input_tensor, Tensor *tensor, int old_height, int old_width, 
           int new_height,int new_width, unsigned int oldstep, unsigned int newstep, 
           unsigned int step, int channels); 

private:
    int m_type;
};





}

#endif
