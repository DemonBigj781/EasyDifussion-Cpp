#!/usr/bin/env python3
import argparse, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DB=json.loads((ROOT/'ci/hardware/gpus.json').read_text())
CUDA_RELEASES=['13.2.2','12.9.2','12.8.1']

def resolve(devices, preferred='auto'):
    devices=list(dict.fromkeys(d for d in devices if d and d!='none'))
    unknown=[d for d in devices if d not in DB]
    if unknown: raise SystemExit('Unknown GPU(s): '+', '.join(unknown))
    cuda=set(); oneapi=set(); rocm=set(); portable=set(); ceiling=99
    for d in devices:
        x=DB[d]
        if 'cuda' in x:
            cuda.add(x['cuda']['sm']+'-real'); ceiling=min(ceiling,x['cuda']['max_major'])
        if 'oneapi' in x: oneapi.add(x['oneapi']['arch'])
        if 'rocm' in x: rocm.add(x['rocm']['arch'])
        portable.update(x.get('portable',[]))
    cuda_version=''
    if cuda:
        pref_major=99 if preferred=='auto' else int(preferred.split('.')[0])
        max_major=min(pref_major,ceiling)
        cuda_version=next((v for v in CUDA_RELEASES if int(v.split('.')[0])<=max_major), '')
        if not cuda_version: raise SystemExit('No compatible CUDA toolkit version in catalog')
    return {'devices':devices,'cuda':sorted(cuda),'cuda_version':cuda_version,'oneapi':sorted(oneapi),'rocm':sorted(rocm),'portable':sorted(portable)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cuda-version',default='auto')
    p.add_argument('--backend',default='unified',choices=['unified','cuda','oneapi','rocm','vulkan','opengl','opencl','directml','cpu'])
    p.add_argument('devices',nargs='*')
    a=p.parse_args(); out=resolve(a.devices,a.cuda_version)
    if a.backend!='unified' and a.backend!='cpu':
        ok=(a.backend=='cuda' and out['cuda']) or (a.backend=='oneapi' and out['oneapi']) or (a.backend=='rocm' and out['rocm']) or (a.backend in out['portable']) or a.backend=='directml'
        if not ok: raise SystemExit(f'Selected hardware has no {a.backend} target')
    print(json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
