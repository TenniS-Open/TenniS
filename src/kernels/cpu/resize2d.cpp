#include <kernels/cpu/resize2d.h>
#include <core/tensor_builder.h>
#include <memory>

namespace ts {


template<typename T>
void Resize2d::ResizeImageLinear( const T *src_im, int src_width, int src_height, int channels,
                                     T *dst_im, int dst_width, int dst_height){
  if (src_width == dst_width && src_height == dst_height) {
    ::memcpy(dst_im, src_im, src_width * src_height * channels * sizeof(T));
    return;
  }

  double lfx_scl = double(src_width) / dst_width;
  double lfy_scl = double(src_height) / dst_height;
  double bias_x = lfx_scl / 2 - 0.5;
  double bias_y = lfy_scl / 2 - 0.5;

  unsigned int srcstep = src_width * src_height;
  unsigned int dststep = dst_width * dst_height;

  for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
    for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {
      double lf_x_s = lfx_scl * n_x_d + bias_x;
      double lf_y_s = lfy_scl * n_y_d + bias_y;

      lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
      lf_x_s = lf_x_s < src_width - 1 ? lf_x_s : src_width - 1 - 1e-5;
      lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
      lf_y_s = lf_y_s < src_height - 1 ? lf_y_s : src_height - 1 - 1e-5;

      int n_x_s = int(lf_x_s);
      int n_y_s = int(lf_y_s);

      double lf_weight_x = lf_x_s - n_x_s;
      double lf_weight_y = lf_y_s - n_y_s;

      for (int c = 0; c < channels; c++) {
          dst_im[(n_y_d * dst_width + n_x_d) * channels + c] =
          (T)((1 - lf_weight_y) * (1 - lf_weight_x) * src_im[(n_y_s * src_width + n_x_s) * channels + c] +
          (1 - lf_weight_y) * lf_weight_x * src_im[(n_y_s * src_width + n_x_s + 1) * channels + c] +
          lf_weight_y * (1 - lf_weight_x) * src_im[((n_y_s + 1) * src_width + n_x_s) * channels + c] +
          lf_weight_y * lf_weight_x * src_im[((n_y_s + 1) * src_width + n_x_s + 1) * channels + c]);

      }//end for c
    }
  }

  return;
}




template<typename T>
 void Resize2d::ResizeImageCubic(const T *src_im, int src_width, int src_height, int channels,
                             T *dst_im, int dst_width, int dst_height){
    double scale_x = (double)src_width / dst_width;
    double scale_y = (double)src_height / dst_height;

    int srcrows = src_width * channels;
    int dstrows = dst_width * channels;
    unsigned int srcstep = src_width * src_height;
    unsigned int dststep = dst_width * dst_height;


    for (int j = 0; j < dst_height; ++j) {
        double fy = (double)((j + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        //sy = std::min(sy, src_height - 3);  
        //sy = std::max(1, sy);  
        if(sy < 1) {
            fy = 0; sy = 1;
        }

        if(sy >= src_height - 3) {
            fy = 0, sy = src_height - 3;
        }

        const double A = -0.75f;

        double coeffsY[4];
        coeffsY[0] = ((A*(fy + 1) - 5*A)*(fy + 1) + 8*A)*(fy + 1) - 4*A;
        coeffsY[1] = ((A + 2)*fy - (A + 3))*fy*fy + 1;
        coeffsY[2] = ((A + 2)*(1 - fy) - (A + 3))*(1 - fy)*(1 - fy) + 1;
        coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

        for (int i = 0; i < dst_width; ++i)
        {
            double fx = (double)((i + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 1) {
               fx = 0, sx = 1;
            }
            if (sx >= src_width - 3) {
               fx = 0, sx = src_width - 3;
            }

            double coeffsX[4];
            coeffsX[0] = ((A*(fx + 1) - 5*A)*(fx + 1) + 8*A)*(fx + 1) - 4*A;
            coeffsX[1] = ((A + 2)*fx - (A + 3))*fx*fx + 1;
            coeffsX[2] = ((A + 2)*(1 - fx) - (A + 3))*(1 - fx)*(1 - fx) + 1;
            coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

            for (int k = 0; k < channels; ++k) {
                dst_im[j * dstrows + i * channels + k] = (T)((src_im [(sy-1) * srcrows + (sx-1) * channels + k] * coeffsX[0] * coeffsY[0] +
                                                         src_im [(sy) * srcrows + (sx-1) * channels + k] * coeffsX[0] * coeffsY[1] +
                                                         src_im [(sy+1) * srcrows + (sx-1) * channels + k] * coeffsX[0] * coeffsY[2] +
                                                         src_im [(sy+2) * srcrows + (sx-1) * channels + k] * coeffsX[0] * coeffsY[3] +

                                                         src_im [(sy-1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[0] +
                                                         src_im [(sy) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[1] +
                                                         src_im [(sy+1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[2] +
                                                         src_im [(sy+2) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[3] +

                                                         src_im [(sy-1) * srcrows + (sx+1) * channels + k] * coeffsX[2] * coeffsY[0] +
                                                         src_im [(sy) * srcrows + (sx+1) * channels + k] * coeffsX[2] * coeffsY[1] +
                                                         src_im [(sy+1) * srcrows + (sx+1) * channels + k] * coeffsX[2] * coeffsY[2] +
                                                         src_im [(sy+2) * srcrows + (sx+1) * channels + k] * coeffsX[2] * coeffsY[3] +

                                                         src_im [(sy-1) * srcrows + (sx+2) * channels + k] * coeffsX[3] * coeffsY[0] +
                                                         src_im [(sy) * srcrows + (sx+2) * channels + k] * coeffsX[3] * coeffsY[1] +
                                                         src_im [(sy+1) * srcrows + (sx+2) * channels + k] * coeffsX[3] * coeffsY[2] +
                                                         src_im [(sy+2) * srcrows + (sx+2) * channels + k] * coeffsX[3] * coeffsY[3] ));

            }//end k
        }
    }
}

///////////////////////////////////////////////////////////
Resize2d:: Resize2d() {
    field("type", OPTIONAL);
    m_type = 0;
}

void Resize2d::init() {
    supper::init();
    if(has("type")) {
        m_type = ts::tensor::to_int(get("type"));
    }

}



int Resize2d::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    if(stack.size() != 2 || stack.index(0)->dims() < 2 || stack.index(0)->dims() != stack.index(1)->count()) {
        throw ts::Exception("Resize2d input parameters is invalid");
    }

    if(stack.index(1)->dtype() != ts::INT32) {
        throw ts::Exception("Resize2d input parameters 1 type only supported INT32");
    }

    int type_len = ts::type_bytes(stack.index(0)->dtype());

    int dims = stack.index(0)->dims();
    int i=0;
    for(i=0; i< dims; i++) {
        if((int)(stack.index(1)->sync(memory_device()).data<int>()[i]) > 0)
            break;
    }

    if(i >= dims - 1) {
        throw ts::Exception("Resize2d input parameters dims invalid");
    }

    int new_height = (int)stack.index(1)->sync(memory_device()).data<int>()[i];
    int new_width  = (int)stack.index(1)->sync(memory_device()).data<int>()[i+1];

    ts::Shape shape(stack.index(0)->sizes());

    shape[i] = new_height;
    shape[i+1] = new_width;

    output.resize(1);
    output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(),shape);
    return 1;
}

template<typename T>
void Resize2d::resize(ts::Stack &stack, ts::Tensor &tensor, int old_height, int old_width,
                      int new_height,int new_width, unsigned int oldstep, unsigned int newstep, 
                      unsigned int step, int channels) {

     T*  psrc = stack.index(0)->sync(memory_device()).data<T>() + oldstep;;
     T*  pdst = tensor.sync(memory_device()).data<T>() + newstep;;

     if(m_type == 0) {
         ResizeImageLinear(psrc, old_width, old_height, channels, pdst,new_width,new_height);
     }else {
         ResizeImageCubic(psrc, old_width, old_height, channels, pdst,new_width,new_height);
     }
}


int Resize2d::run(ts::Stack &stack) {
    int input_num = stack.size();

    if(input_num != 2 || stack.index(0)->dims() < 2 || stack.index(0)->dims() != stack.index(1)->count()) {
        throw ts::Exception("Resize2d input parameters is invalid");
    }

    if(stack.index(1)->dtype() != ts::INT32) {
        throw ts::Exception("Resize2d input parameters 1 type only supported INT32");
    }
    //int type_len = ts::type_bytes(stack.index(0)->dtype());

    int dims = stack.index(0)->dims();
    int i=0;
    for(i=0; i< dims; i++) {
        if((int)(stack.index(1)->sync(memory_device()).data<int>()[i]) > 0)
            break;
    }

    if(i >= dims - 1) {
         throw ts::Exception("Resize2d input parameters dims invalid");
    }

    int new_height = (int)stack.index(1)->sync(memory_device()).data<int>()[i];
    int new_width  = (int)stack.index(1)->sync(memory_device()).data<int>()[i+1];
    int old_height = (int)stack.index(0)->sizes()[i];
    int old_width  = (int)stack.index(0)->sizes()[i+1];

    std::vector<ts::Tensor::Prototype> output;
    infer(stack, output);

    ts::Shape shape(output[0].sizes());

    int ntotalbuffer = 0;
    int batchs,channels;
    batchs = channels = 1;

    for(int k=0; k<i; k++) {
        batchs *= shape[k];
    }

    for(int k=i+2; k<shape.size(); k++) {
        channels *= shape[k];
    }

    stack.push(output[0], memory_device());

    int newstep = channels * new_height * new_width;
    int oldstep = channels * old_height * old_width;

    auto &tensor = *stack.index(-1);
    ts::DTYPE type = tensor.dtype();

    for(int k=0; k<batchs; k++) {
        if(type == ts::UINT8){
            resize<unsigned char>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep, newstep, channels);
        }else if(type == ts::INT8) {
            resize<char>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep,  newstep, channels);
        }else if(type == ts::INT16) {
            resize<short>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep,  newstep, channels);
        }else if(type == ts::UINT16) {
            resize<unsigned short>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep,  newstep, channels);
        }else if(type == ts::INT32) {
            resize<int>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep,  newstep, channels);
        }else if(type == ts::UINT32) {
            resize<unsigned int>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep, newstep, channels);
        }else if(type == ts::FLOAT32) {
            resize<float>(stack, tensor, old_height,old_width,new_height,new_width,
                                     k * oldstep, k * newstep, newstep, channels);
        }else if(type == ts::FLOAT64) {
            resize<double>(stack, tensor, old_height,old_width,new_height,new_width,
                                   k * oldstep, k * newstep, newstep, channels);
        }else {
            throw ts::Exception("Resize2d do not support data type");
        }

    }

    return 1;
}






//////////////////////////////////////////////////////////

TS_REGISTER_OPERATOR(Resize2d, ts::CPU, "_resize2d")

}
