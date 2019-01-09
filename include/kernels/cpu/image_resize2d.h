#ifndef TS_KERNELS_IMAGE_RESIZE2D_H
#define TS_KERNELS_IMAGE_RESIZE2D_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
#include <math.h>



namespace ts {


//support UINT8,UINT16,FLOAT32,FLOAT64, channel 1,2,3
class Image_Resize2d : public ts::Operator {
public:


    template<typename T>
    static void ResizeImageLinear( const T *src_im, int src_width, int src_height, int channels,
                                     T *dst_im, int dst_width, int dst_height, bool ishwc);

    template<typename T>
    static void ResizeImageCubic(const T *src_im, int src_width, int src_height, int channels,
                             T *dst_im, int dst_width, int dst_height, bool ishwc);

public:
    enum RESIZETYPE{
        linear,
        cublic
    };

    using supper = ts::Operator;
    Image_Resize2d();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

private:
template<typename T>
    void resize(ts::Stack &stack, ts::Tensor &tensor, int old_height, int old_width, 
           int new_height,int new_width, unsigned int oldstep, unsigned int newstep, unsigned int step, int channels, bool ishwc); 


private:
    int m_type;
};

}

#endif
