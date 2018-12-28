#include <kernels/cpu/resize2d.h>
#include <kernels/cpu/saturate_cast.h>
#include <memory>

namespace ts {


template<typename T>
void Resize2d::ResizeImageLinear( const T *src_im, int src_width, int src_height, int channels,
                                     T *dst_im, int dst_width, int dst_height, bool ishwc){
  if ((channels != 1 && channels != 3)) {
    std::cout << "<Illegal image channels!>" << std::endl;
    std::cout << "src_img: " << channels << std::endl;
    throw ts::Exception("input channels only support 1 and 3");
  }

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
          if(ishwc) {
              dst_im[(n_y_d * dst_width + n_x_d) * channels + c] =
                     (1 - lf_weight_y) * (1 - lf_weight_x) * src_im[(n_y_s * src_width + n_x_s) * channels + c] +
                     (1 - lf_weight_y) * lf_weight_x * src_im[(n_y_s * src_width + n_x_s + 1) * channels + c] +
                     lf_weight_y * (1 - lf_weight_x) * src_im[((n_y_s + 1) * src_width + n_x_s) * channels + c] +
                     lf_weight_y * lf_weight_x * src_im[((n_y_s + 1) * src_width + n_x_s + 1) * channels + c];
          }else {
              dst_im[(n_y_d * dst_width + n_x_d) + c * dststep ] =
                     (1 - lf_weight_y) * (1 - lf_weight_x) * src_im[(n_y_s * src_width + n_x_s) + c * srcstep] +
                     (1 - lf_weight_y) * lf_weight_x * src_im[(n_y_s * src_width + n_x_s + 1) + c * srcstep] + 
                     lf_weight_y * (1 - lf_weight_x) * src_im[((n_y_s + 1) * src_width + n_x_s) + c * srcstep] +
                     lf_weight_y * lf_weight_x * src_im[((n_y_s + 1) * src_width + n_x_s + 1) + c * srcstep];

         }//end if
      }//end for c
    }
  }

  return;
}




template<typename T>
 void Resize2d::ResizeImageCubic(const T *src_im, int src_width, int src_height, int channels,
                             T *dst_im, int dst_width, int dst_height, bool ishwc){
    double scale_x = (double)src_width / dst_width;
    double scale_y = (double)src_height / dst_height;

    int srcrows = src_width * channels;
    int dstrows = dst_width * channels;
    unsigned int srcstep = src_width * src_height;
    unsigned int dststep = dst_width * dst_height;


    for (int j = 0; j < dst_height; ++j) {
        float fy = (float)((j + 0.5) * scale_y - 0.5);
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

        const float A = -0.75f;

        float coeffsY[4];
        coeffsY[0] = ((A*(fy + 1) - 5*A)*(fy + 1) + 8*A)*(fy + 1) - 4*A;
        coeffsY[1] = ((A + 2)*fy - (A + 3))*fy*fy + 1;
        coeffsY[2] = ((A + 2)*(1 - fy) - (A + 3))*(1 - fy)*(1 - fy) + 1;
        coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

        short cbufY[4];
        cbufY[0] = ts::saturate_cast<short>(coeffsY[0] * 2048);
        cbufY[1] = ts::saturate_cast<short>(coeffsY[1] * 2048);
        cbufY[2] = ts::saturate_cast<short>(coeffsY[2] * 2048);
        cbufY[3] = ts::saturate_cast<short>(coeffsY[3] * 2048);

        for (int i = 0; i < dst_width; ++i)
        {
            float fx = (float)((i + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 1) {
               fx = 0, sx = 1;
            }
            if (sx >= src_width - 3) {
               fx = 0, sx = src_width - 3;
            }

            float coeffsX[4];
            coeffsX[0] = ((A*(fx + 1) - 5*A)*(fx + 1) + 8*A)*(fx + 1) - 4*A;
            coeffsX[1] = ((A + 2)*fx - (A + 3))*fx*fx + 1;
            coeffsX[2] = ((A + 2)*(1 - fx) - (A + 3))*(1 - fx)*(1 - fx) + 1;
            coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

            short cbufX[4];
            cbufX[0] = ts::saturate_cast<short>(coeffsX[0] * 2048);
            cbufX[1] = ts::saturate_cast<short>(coeffsX[1] * 2048);
            cbufX[2] = ts::saturate_cast<short>(coeffsX[2] * 2048);
            cbufX[3] = ts::saturate_cast<short>(coeffsX[3] * 2048);

            for (int k = 0; k < channels; ++k) {
                if(ishwc) {
                    dst_im[j * dstrows + i * channels + k] = abs(int((src_im [(sy-1) * srcrows + (sx-1) * channels + k] * cbufX[0] * cbufY[0] +
                                                                  src_im [(sy) * srcrows + (sx-1) * channels + k] * cbufX[0] * cbufY[1] +
                                                                  src_im [(sy+1) * srcrows + (sx-1) * channels + k] * cbufX[0] * cbufY[2] +
                                                                  src_im [(sy+2) * srcrows + (sx-1) * channels + k] * cbufX[0] * cbufY[3] +

                                                                  src_im [(sy-1) * srcrows + (sx) * channels + k] * cbufX[1] * cbufY[0] +
                                                                  src_im [(sy) * srcrows + (sx) * channels + k] * cbufX[1] * cbufY[1] +

                                                                  src_im [(sy+1) * srcrows + (sx) * channels + k] * cbufX[1] * cbufY[2] +
                                                                  src_im [(sy+2) * srcrows + (sx) * channels + k] * cbufX[1] * cbufY[3] +

                                                                  src_im [(sy-1) * srcrows + (sx+1) * channels + k] * cbufX[2] * cbufY[0] +
                                                                  src_im [(sy) * srcrows + (sx+1) * channels + k] * cbufX[2] * cbufY[1] +

                                                                  src_im [(sy+1) * srcrows + (sx+1) * channels + k] * cbufX[2] * cbufY[2] +
                                                                  src_im [(sy+2) * srcrows + (sx+1) * channels + k] * cbufX[2] * cbufY[3] +

                                                                  src_im [(sy-1) * srcrows + (sx+2) * channels + k] * cbufX[3] * cbufY[0] +
                                                                  src_im [(sy) * srcrows + (sx+2) * channels + k] * cbufX[3] * cbufY[1] +


                                                                  src_im [(sy+1) * srcrows + (sx+2) * channels + k] * cbufX[3] * cbufY[2] +
                                                                  src_im [(sy+2) * srcrows + (sx+2) * channels + k] * cbufX[3] * cbufY[3] )) >> 22);

                }else {
                    dst_im[j * dst_width + i + k * dststep] = abs(int((src_im [(sy-1) * src_width + (sx-1) + k * srcstep] * cbufX[0] * cbufY[0] +
                                                                  src_im [(sy) * src_width + (sx-1) + k * srcstep] * cbufX[0] * cbufY[1] +
                                                                  src_im [(sy+1) * src_width + (sx-1) + k * srcstep] * cbufX[0] * cbufY[2] +
                                                                  src_im [(sy+2) * src_width + (sx-1) + k * srcstep] * cbufX[0] * cbufY[3] +

                                                                  src_im [(sy-1) * src_width + (sx) + k * srcstep] * cbufX[1] * cbufY[0] +
                                                                  src_im [(sy) * src_width + (sx)  + k * srcstep] * cbufX[1] * cbufY[1] +

                                                                  src_im [(sy+1) * src_width + (sx) + k * srcstep] * cbufX[1] * cbufY[2] +
                                                                  src_im [(sy+2) * src_width + (sx) + k * srcstep] * cbufX[1] * cbufY[3] +

                                                                  src_im [(sy-1) * src_width + (sx+1) + k * srcstep] * cbufX[2] * cbufY[0] +
                                                                  src_im [(sy) * src_width + (sx+1) + k * srcstep] * cbufX[2] * cbufY[1] +

                                                                  src_im [(sy+1) * src_width + (sx+1) + k * srcstep] * cbufX[2] * cbufY[2] +
                                                                  src_im [(sy+2) * src_width + (sx+1) + k * srcstep] * cbufX[2] * cbufY[3] +

                                                                  src_im [(sy-1) * src_width + (sx+2) + k * srcstep] * cbufX[3] * cbufY[0] +
                                                                  src_im [(sy) * src_width + (sx+2)  + k * srcstep] * cbufX[3] * cbufY[1] +


                                                                  src_im [(sy+1) * src_width + (sx+2) + k * srcstep] * cbufX[3] * cbufY[2] +
                                                                  src_im [(sy+2) * src_width + (sx+2) + k * srcstep] * cbufX[3] * cbufY[3] )) >> 22);

                }//end if

            }//end k
        }
    }
}



TS_REGISTER_OPERATOR(Resize2d, ts::CPU, "_resize2d")

}
