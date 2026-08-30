# Build and release
```
git clone https://github.com/easydiffusion/sdkit.git
git checkout v3
git submodule init
git submodule update --recursive

python -m scripts.build_all
python -m scripts.upload_all --tag v3.0.0
```

# Set up the runners

## Windows

1. Download and run ["Build Tools for Visual Studio"](https://visualstudio.microsoft.com/downloads/)
2. Install or Edit:
 - MSVC - VS 2022 C+ x64/x86 build tools
 - MSVC - VS 2022 C+ ARM64 build tools
 - C++ Clang Compiler for Windows
 - MSBuild support for LLVM (clang-cl) toolset
 - C++ CMake tools for Windows
3. Install [Ninja](https://github.com/ninja-build/ninja/releases) and make it available on `PATH`.


## Linux
1. Download a `runfile (local)` from [CUDA Toolkit downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04).
2. Download an SDK Tarball from [Vulkan SDK downloads](https://vulkan.lunarg.com/sdk/home).

### Non-Linux host (using Docker)
```bash
docker volume create ed-build
docker run -it --gpus=all -v ed-build:/ed -v /path/to/models:/models --name ed-build ubuntu:latest

# in a different tab
docker cp vulkansdk-linux-x86_64-1.4.328.1.tar.xz ed-build:/ed/
docker cp cuda_12.8.1_570.124.06_linux.run ed-build:/ed/
```

### Inside Linux
```bash
apt update
apt install git curl vim python-is-python3 python3 xz-utils -y
apt install g++ ccache cmake -y

# install ninja
cd
mkdir bin
wget https://github.com/ninja-build/ninja/releases/download/latest/ninja-linux.zip
unzip ninja-linux.zip
mv ninja bin/

# install CUDA Toolkit
cd /ed
./cuda_12.8.1_570.124.06_linux.run
echo "export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/lib64:\$PATH" >> ~/.bashrc

# install Vulkan SDK
cd /ed
tar -xf vulkansdk-linux-x86_64-1.4.328.1.tar.xz
mv 1.4.328.1 vulkan-1.4.328.1
echo "source /ed/vulkan-1.4.328.1/setup-env.sh" >> ~/.bashrc
apt install vulkan-tools vulkan-validationlayers -y
```
