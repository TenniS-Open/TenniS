#include <vector>
#include <iostream>

#include <kernels/cpu/im2col.h>
#include <cstring>


namespace ts {

template <typename Dtype>
void tensorstack_set(const int N, const Dtype alpha, Dtype* Y) {
	if (alpha == 0) {
		memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
		return;
	}
	for (int i = 0; i < N; ++i) {
		Y[i] = alpha;
	}
}


// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h_top, const int pad_h_bottom, const int pad_w_left,const int pad_w_right,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col, const Dtype padding_value) {
  const int output_h = (height +  pad_h_top + pad_h_bottom -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;

  const int output_w = (width +  pad_w_left + pad_w_right -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h_top + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) 
		{
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = padding_value;//0;
            }
          } else {
            int input_col = -pad_w_left + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = padding_value;//0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h_top, const int pad_h_bottom, const int pad_w_left,const int pad_w_right, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col, const float padding_value);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h_top, const int pad_h_bottom, const int pad_w_left,const int pad_w_right, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col, const double padding_value);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  tensorstack_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;

  //auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
  //if (gun == nullptr || gun->size() <= 1) {
	  for (int channel = channels; channel--; data_im += channel_size) {
		  for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			  for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				  int input_row = -pad_h + kernel_row * dilation_h;
				  for (int output_rows = output_h; output_rows; output_rows--) {
					  if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						  data_col += output_w;
					  }
					  else {
						  int input_col = -pad_w + kernel_col * dilation_w;
						  for (int output_col = output_w; output_col; output_col--) {
							  if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								  data_im[input_row * width + input_col] += *data_col;
							  }
							  data_col++;
							  input_col += stride_w;
						  }
					  }
					  input_row += stride_h;
				  }
			  }
		  }
	  }
  //}
/*
  else
  {
	  auto col_size = kernel_h * kernel_w * output_h * output_w;
	  auto channel_bins = orz::split_bins(0, channels, int(gun->size()));
	  for (auto &channel_bin : channel_bins)
	  {
		  gun->fire([&, channel_bin](int) {
			  auto local_data_im = data_im + channel_bin.first * channel_size;
			  auto local_data_col = data_col + channel_bin.first * col_size;
			  for (int channel = channel_bin.first; channel < channel_bin.second; ++channel, local_data_im += channel_size) {
				  for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
					  for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
						  int input_row = -pad_h + kernel_row * dilation_h;
						  for (int output_rows = output_h; output_rows; output_rows--) {
							  if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
								  local_data_col += output_w;
							  }
							  else {
								  int input_col = -pad_w + kernel_col * dilation_w;
								  for (int output_col = output_w; output_col; output_col--) {
									  if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
										  local_data_im[input_row * width + input_col] += *local_data_col;
									  }
									  local_data_col++;
									  input_col += stride_w;
								  }
							  }
							  input_row += stride_h;
						  }
					  }
				  }
			  }
		  });
	  }
	  gun->join();
  }
  */
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

}