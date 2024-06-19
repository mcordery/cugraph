To build with cmake first configure the build

```zsh
# This is for hippified cugraph
cmake -S cpp -B cpp/build \
    -DCMAKE_C_COMPILER=/opt/rocm-6.1.2/bin/hipcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm-6.1.2/bin/hipcc \
    -DUSE_CUGRAPH_OPS=OFF \
    -DBUILD_CUGRAPH_MG_TESTS=OFF \
    -DBUILD_CUGRAPH_MTMG_TESTS=OFF \
    -DUSE_HIP=ON

# This is for hippified cugraph on thr ROCm 6.3 container
cmake -S cpp -B cpp/build \
    -DCMAKE_C_COMPILER=/opt/rocm-6.3.0-14269/bin/hipcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.0-14269/bin/hipcc \
    -DUSE_CUGRAPH_OPS=OFF \
    -DBUILD_CUGRAPH_MG_TESTS=OFF \
    -DBUILD_CUGRAPH_MTMG_TESTS=OFF \
    -DUSE_HIP=ON
```

and build with 6 parallel threads
```zsh
cmake --build cpp/build --parallel 6 2>&1 | tee build.log
```