# Dependencies
* [CCCL tag v2.2.0](https://github.com/NVIDIA/cccl/tree/v2.2.0)
* OS level google test package. On ubuntu that will be `apt install libgtest-dev`.


To build with cmake first configure the build

```zsh
cmake -S cpp -B cpp/build \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc \
    -DUSE_CUGRAPH_OPS=OFF \
    -DBUILD_CUGRAPH_MG_TESTS=OFF \
    -DBUILD_CUGRAPH_MTMG_TESTS=OFF \
    -DUSE_CUDA=ON
```

and build with 6 parallel threads
```zsh
cmake --build cpp/build --parallel 6
```
