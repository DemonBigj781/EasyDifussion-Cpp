#!/usr/bin/env python3
import argparse, json, platform, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'source/sdkit3-port-source'
def run(*cmd): print('+',' '.join(map(str,cmd)),flush=True); subprocess.run(list(map(str,cmd)),check=True)
def configure(backend,target,mode,suffix=''):
    system=platform.system().lower(); b=ROOT/'build'/f'ci-{system}-{backend}{suffix}'
    if system=='windows':
        if backend!='cpu': raise SystemExit(f'{backend} Windows provisioning is not wired yet')
        args=['cmake','-S',SRC,'-B',b,'-A','x64','-DSDKIT_BUILD_LLAMA_RUNTIME=OFF','-DSDKIT_BUILD_NATIVE_VISION=OFF','-DGGML_CCACHE=OFF']
    else:
        args=['cmake','-S',SRC,'-B',b,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSDKIT_BUILD_LLAMA_RUNTIME=OFF','-DSDKIT_BUILD_NATIVE_VISION=OFF','-DSDKIT_OPENGL_BACKEND=OFF','-DGGML_CCACHE=OFF','-DSD_CUDA=OFF','-DSD_HIPBLAS=OFF','-DSD_METAL=OFF','-DSD_VULKAN=OFF','-DSD_OPENCL=OFF','-DSD_SYCL=OFF','-DSD_MUSA=OFF']
        if backend=='cuda': args += ['-DSD_CUDA=ON',f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(target['cuda'])}"]
        elif backend=='rocm': args += ['-DSD_HIPBLAS=ON',f"-DGPU_TARGETS={';'.join(target['rocm'])}",f"-DAMDGPU_TARGETS={';'.join(target['rocm'])}"]
        elif backend=='vulkan': args += ['-DSD_VULKAN=ON']
        elif backend=='opencl': args += ['-DSD_OPENCL=ON']
        elif backend=='opengl': args += ['-DSDKIT_OPENGL_BACKEND=ON']
        elif backend=='oneapi':
            arch=target['_oneapi_arch']; args += ['-DCMAKE_C_COMPILER=icx','-DCMAKE_CXX_COMPILER=icpx','-DGGML_OPENMP=OFF','-DSD_SYCL=ON','-DGGML_SYCL_TARGET=INTEL','-DGGML_SYCL_F16=ON','-DGGML_SYCL_DNN=ON','-DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON']
            if arch!='jit': args += [f'-DGGML_SYCL_DEVICE_ARCH={arch}']
        elif backend=='directml': raise SystemExit('DirectML/D3D12 implementation is not wired yet')
    run(*args)
    if mode in ('compile','compile-and-regression'): run('cmake','--build',b,'--target','sdkit','--config','Release','--parallel','2')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--backend',required=True); p.add_argument('--mode',default='compile'); p.add_argument('--target-json',required=True); a=p.parse_args(); t=json.loads(a.target_json)
    if a.backend=='oneapi':
        for arch in t['oneapi']:
            x=dict(t); x['_oneapi_arch']=arch; configure('oneapi',x,a.mode,'-'+arch)
    else: configure(a.backend,t,a.mode)
if __name__=='__main__': main()
