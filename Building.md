# Dependencies
* [CCCL tag v2.2.0](https://github.com/NVIDIA/cccl/tree/v2.2.0)
* OS level google test package. On ubuntu that will be `apt install libgtest-dev`.


To build with cmake first configure the build

```zsh
# This is for cugraph
cmake -S cpp -B cpp/build \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
    -DUSE_CUGRAPH_OPS=OFF \
    -DBUILD_CUGRAPH_MG_TESTS=OFF \
    -DBUILD_CUGRAPH_MTMG_TESTS=OFF \
    -DUSE_CUDA=ON

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

To build in a container spin up a docker container on cgy-geodude.
This currently does not seem to work. It explodes when hitting the first memory allocation probably because it can't access the hardware from the container. 

```zsh
docker run -ti --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-devel-ubuntu22.04
```

Then inside the container

```bash
apt update
apt install \
nano \
wget \
software-properties-common \
cuda-toolkit-12-4 \
libgtest-dev \
git \
htop \
tmux \
g++-11

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository "deb https://apt.kitware.com/ubuntu/ jammy main"
apt update
apt install cmake

mkdir ~/.ssh;
echo -e "<body of ed25519 private key with github access>" > ~/.ssh/id_ed25519

chmod 600 ~/.ssh/id_ed25519

echo -e "[user]
	email = <github login>" > ~/.gitconfig

```


https://compute-artifactory.amd.com/artifactory/rocm-docker-experimental/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang/
https://confluence.amd.com/display/MLSE/Using+Docker+containers+from+compute-artifactory
http://rocm-dockers-list:8506/