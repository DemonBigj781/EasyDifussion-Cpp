#!/usr/bin/env python3
"""Build/run policy units with the real device selector and an explicit mock factory.

This never loads Vulkan, executes WebGPU kernels, or verifies physical hardware.
Build once on Linux, then run the identical artifact on both target systems.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parents[1]
CASES = ('hardware-first', 'embedded-fallback', 'native-requires-hardware',
         'explicit-native-software', 'missing-and-ambiguous', 'configured-default',
         'immutable-factory', 'native-thread-guard')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--sdk', type=Path, default=HERE / 'out/sdk')
    parser.add_argument('--foundation', type=Path, default=HERE / 'out/software-webgpu/foundation')
    parser.add_argument('--out', type=Path, default=HERE / 'out/device-policy-test')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--loader', type=Path)
    args = parser.parse_args()
    out, sdk, foundation = args.out.resolve(), args.sdk.resolve(), args.foundation.resolve()
    artifact = args.probe.resolve() if args.probe else out / 'device-policy-test.exe'
    loader = args.loader.resolve() if args.loader else None
    if args.build:
        spec = importlib.util.spec_from_file_location('cosmo_build', HERE / 'build.py')
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        if (sdk / 'ARCHIVE.sha256').read_text().strip() != build.SDK_SHA256:
            raise RuntimeError('Policy probe requires the pinned application SDK')
        out.mkdir(parents=True, exist_ok=True)
        upstream = foundation / 'third_party/wgpu_native/upstream'
        sources = [HERE / 'backend/wgpu_devices.cpp', HERE / 'tests/device_policy_test.cpp']
        includes = [HERE / 'backend/include', foundation / 'third_party/wgpu_native/runtime',
                    foundation / 'third_party/lavapipe', upstream / 'ffi', upstream / 'ffi/webgpu-headers']
        debug = out / 'device-policy-test.com.dbg'
        commands = [
            [sdk / 'bin/x86_64-unknown-cosmo-c++', '-std=c++20', '-O1', '-Wall', '-Wextra', '-Werror', '-pthread',
             *sources, *['-I' + str(x) for x in includes], '-o', debug],
            [sdk / 'bin/ape-x86_64.elf', sdk / 'bin/apelink', '-o', artifact, '-l', sdk / 'bin/ape-x86_64.elf', debug],
        ]
        for command in commands:
            subprocess.run([str(x) for x in command], check=True)
        build.set_stack(artifact)
        loader = out / 'ape-x86_64.elf'
        shutil.copy2(sdk / 'bin/ape-x86_64.elf', loader)
        headers = [upstream / 'ffi/wgpu.h', upstream / 'ffi/webgpu-headers/webgpu.h', HERE / 'backend/include/cosmo-webgpu.h']
        (out / 'BUILD.json').write_text(json.dumps({'scope': 'policy unit with mocked factory',
            'sdk_sha256': build.SDK_SHA256, 'commands': [[str(x) for x in c] for c in commands],
            'sources_sha256': {str(p): sha(p) for p in sources + headers}, 'artifact_sha256': sha(artifact)}, indent=2) + '\n')
    before = sha(artifact)
    observations = []
    for scenario in CASES:
        command = ([str(loader)] if loader else []) + [str(artifact), scenario]
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f'{scenario}: status {result.returncode}\n{result.stdout}\n{result.stderr}')
        value = json.loads(result.stdout)
        if value.get('status') != 'PASS' or value.get('scenario') != scenario:
            raise RuntimeError(f'{scenario}: invalid evidence {value}')
        observations.append(value)
    if sha(artifact) != before:
        raise RuntimeError('Policy test artifact changed during execution')
    print(json.dumps({'status': 'PASS', 'scope': 'device policy unit; factory mocked; no Vulkan/GPU execution',
                      'artifact_sha256': before, 'cases': observations}, indent=2))


if __name__ == '__main__':
    main()
