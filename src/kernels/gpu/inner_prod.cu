#include <kernels/gpu/inner_prod.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>



namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void gpu_inner_prod_compute_run_kernel(int m, int n, int k, const T *A, const T *B, T *C) {
	    __shared__ T ds_A[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM]; 
	    __shared__ T ds_B[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
 
	    int bx = blockIdx.x;	
  	    int by = blockIdx.y;
	    int tx = threadIdx.x;
	    int ty = threadIdx.y;
 
	    int Row = by * TRANS_BLOCK_DIM + ty;
	    int Col = bx * TRANS_BLOCK_DIM + tx;

            T comp = 0;
	    T Cvalue = 0;
 
	    for (int t=0; t<(n-1)/TRANS_BLOCK_DIM+1; ++t) {
		if (Row < m && t * TRANS_BLOCK_DIM + tx < n)	
		    ds_A[tx][ty] = A[Row*n+t*TRANS_BLOCK_DIM+tx];
		else
		    ds_A[tx][ty] = 0.0;
 
		if (t * TRANS_BLOCK_DIM + ty < n && Col < k)
                    ds_B[tx][ty] = B[(t*TRANS_BLOCK_DIM + ty)*k+Col];
		else
		    ds_B[tx][ty] = 0.0;	
 
		__syncthreads();
		
		for (int i = 0; i < TRANS_BLOCK_DIM; ++i) {
                    //Cvalue += ds_A[i][ty] * ds_B[tx][i];
                    T t;
                    comp -= ds_A[i][ty] * ds_B[tx][i];
                    t = Cvalue - comp;
                    comp = (t - Cvalue) + comp;
                    Cvalue = t;
                }
 
		__syncthreads();
 
		if(Row < m && Col < k) {
	    	    C[Row*k+Col]=Cvalue;
                }		
	    }//end for
        }



        template<typename T>
        static void gpu_inner_prod_compute_run(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            const Shape &lhs_shape = lhs.sizes();
            const Shape &rhs_shape = rhs.sizes();
            // const Shape &out_shape = out.sizes();

            const T *psrc = lhs.data<T>();
            const T *pdot = rhs.data<T>();
            T *pdst = out.data<T>();

            //std::cout << "m:" << lhs_shape[0] << ",n:" << lhs_shape[1] << ",k:" << rhs_shape[1] << ",:" << rhs_shape[0] << std::endl;

            dim3 blocksize(CUDA_BLOCK(rhs_shape[1], TRANS_BLOCK_DIM), CUDA_BLOCK(lhs_shape[0], TRANS_BLOCK_DIM),1);
            dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM,1);

            //dim3 dimGrid((rhs_shape[1]-1)/TILE_WIDTH+1,(lhs_shape[0]-1)/TILE_WIDTH+1,1); 
            //dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);


            gpu_inner_prod_compute_run_kernel<T> <<<blocksize, threadsize>>> (lhs_shape[0], lhs_shape[1], rhs_shape[1], psrc, pdot, pdst);

        }

        void InnerProd::inner_prod(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_inner_prod_compute_run<TYPE>(lhs, rhs, out); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(InnerProd, GPU, name::layer::inner_prod())
