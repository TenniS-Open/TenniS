# TensorStack 

## Compilation

The default installation DIR is ` ${PROJECT_BINARY_DIR}/build`,
add `-DCMAKE_INSTALL_PREFIX` to change installation DIR.

1. Do `cmake` or `CPU`:
```
cmake ..
-DTS_USE_CBLAS=ON
-DTS_USE_OPENMP=ON
-DTS_USE_SIMD=ON
-DTS_USE_AVX=ON
-DTS_USE_FAST_MATH=OFF
-DTS_BUILD_TEST=OFF
-DTS_BUILD_TOOLS=OFF
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=/usr/local
```
or `GPU`:
```
cmake ..
-DTS_USE_CUDA=ON
-DTS_USE_CUBLAS=ON
-DTS_USE_CBLAS=ON
-DTS_USE_OPENMP=ON
-DTS_USE_SIMD=ON
-DTS_USE_AVX=ON
-DTS_USE_FAST_MATH=OFF
-DTS_BUILD_TEST=OFF
-DTS_BUILD_TOOLS=OFF
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=/usr/local
```

2. Do `make -j16` and waiting.

3. Do `make install`.

4. Find headers and libraries in `CMAKE_INSTALL_PREFIX`