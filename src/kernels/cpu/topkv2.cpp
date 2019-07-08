#include "kernels/cpu/topkv2.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>

namespace ts {
    namespace cpu {
        template <typename T>
        static void adjust_node(T *arr, int n, int len, int *arr2) {
            int l, r, max, index;
            T tmp;
            l = 2 * n + 1; 
            r = 2 * n + 2;
            max = n;

            if (l<len&&arr[l]>arr[n])
                max = l;
            if (r<len&&arr[r]>arr[max])
                max = r;
    
            if (max != n) {
                tmp = arr[n];
                arr[n] = arr[max];
                arr[max] = tmp;

                index = arr2[n];
                arr2[n] = arr2[max];
                arr2[max] = index; 
                adjust_node(arr, max, len, arr2);
            }
        }

        template <typename T>
        static void sort_heap(T *arr, int len, int *arr2) {
            for (int i = len / 2; i >= 0; i--)
                adjust_node(arr, i, len, arr2);
            int index;
            T   tmp;
            for (int i = len - 1; i >= 0; i--) {
                tmp = arr[0];
                arr[0] = arr[i];
                arr[i] = tmp;

                index = arr2[0];
                arr2[0] = arr2[i];
                arr2[i] = index;
                adjust_node(arr, 0, i, arr2);
            }
        }


        template <typename T>
        static void cpu_topkv2_compute_run(const Tensor &x, const int m_number, const int m_sorted, Tensor &out) {
            auto &x_shape = x.sizes();

            T * p_outdata = out.data<T>();
            const T* p_xdata  = x.data<T>();

            Shape out_shape = out.sizes();
            
            Tensor sort_tensor(out.device(), INT32, out_shape);
         
            int * psort = sort_tensor.data<int>(); 
            int number = out.count();
            int steps = number / out_shape[out_shape.size() - 1];
            int out_stride = out_shape[out_shape.size() - 1];
            int x_stride = x_shape[x_shape.size() - 1];

            for(int k=0; k< steps; k++) {
                ::memcpy(p_outdata + k * out_stride, p_xdata + k * x_stride, out_stride * sizeof(T));
                for(int i=0; i<out_stride; i++) {
                    psort[i + k * out_stride] = i;
                } 

                sort_heap<T>(p_outdata + k * out_stride, out_stride, psort + k * out_stride);
                
                for(int i=out_stride; i<x_stride; i++) {
                    if(p_xdata[i + k * x_stride ] < p_outdata[ k * out_stride]) {
                        continue;
                    }

                    p_outdata[k * out_stride] = p_xdata[i + k * x_stride];
                    psort[k * out_stride] = i;
                    sort_heap<T>(p_outdata + k * out_stride, out_stride, psort + k * out_stride); 
                }
                     
            }

            std::vector<Tensor> fields;
            fields.push_back(out);
            fields.push_back(sort_tensor); 
            out.pack(fields);

        }


        void Topkv2::topkv2(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_topkv2_compute_run<TYPE>(x, m_number, m_sorted, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }

        }

    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Topkv2, CPU, name::layer::topkv2())
