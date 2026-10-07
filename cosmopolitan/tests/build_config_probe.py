#!/usr/bin/env python3
"""Build one portable configuration test artifact using the existing pinned SDK."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=HERE / 'out/sdk')
    parser.add_argument('--out', type=Path, default=HERE / 'out/config-test')
    args = parser.parse_args()
    sdk, out = args.sdk.resolve(), args.out.resolve()
    spec = importlib.util.spec_from_file_location('cosmo_build', HERE / 'build.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    if (sdk / 'ARCHIVE.sha256').read_text().strip() != build.SDK_SHA256:
        raise RuntimeError('Configuration probe requires the pinned application SDK')
    out.mkdir(parents=True, exist_ok=True)
    debug, artifact = out / 'config-probe.com.dbg', out / 'config-probe.exe'
    sources = [HERE / 'src/config.cpp', HERE / 'tests/config_probe.cpp']
    commands = [
        [sdk / 'bin/x86_64-unknown-cosmo-c++', '-std=c++20', '-O1', '-Wall', '-Wextra', '-Werror', '-pthread',
         *sources, '-I' + str(HERE / 'src'),
         '-I' + str(HERE.parent / 'source/sdkit3-port-source/stable-diffusion.cpp/thirdparty'), '-o', debug],
        [sdk / 'bin/ape-x86_64.elf', sdk / 'bin/apelink', '-o', artifact, '-l', sdk / 'bin/ape-x86_64.elf', debug],
    ]
    for command in commands:
        subprocess.run([str(value) for value in command], check=True)
    build.set_stack(artifact)
    shutil.copy2(sdk / 'bin/ape-x86_64.elf', out / 'ape-x86_64.elf')
    metadata = {'sdk_sha256': build.SDK_SHA256, 'commands': [[str(x) for x in c] for c in commands],
                'sources': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'artifact': {'file': artifact.name, 'bytes': artifact.stat().st_size,
                             'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    (out / 'BUILD.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['artifact']))


if __name__ == '__main__':
    main()
