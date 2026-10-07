#!/usr/bin/env python3
"""Exercise the actual portable shell with a small, explicitly fake app dispatcher.

Build once using --build on Linux, then run the same shell-probe.exe on Windows.
This does not initialize the production application, HTTP, a model or a driver.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

PORT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_path(value):
    # Cosmopolitan reports /C/... on Windows. Python may expand RUNNER~1 to
    # its long spelling while Cosmo retains it, so compare filesystem identity
    # below rather than assuming that two correct spellings are equal.
    if os.name == 'nt' and len(value) >= 3 and value[0] == '/' and value[1].isalpha() and value[2] == '/':
        value = value[1] + ':' + value[2:]
    return Path(value)


def build_probe(sdk, output):
    spec = importlib.util.spec_from_file_location('cosmo_shell_probe_build', PORT / 'build.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    require((sdk / 'ARCHIVE.sha256').read_text().strip() == build.SDK_SHA256, 'Pinned SDK identity differs')
    output.mkdir(parents=True, exist_ok=True)
    # A fresh package destination avoids depending on incremental apelink behavior.
    stage = Path(tempfile.mkdtemp(prefix='shell-probe-link-', dir=output))
    sources = [PORT / 'src/shell.c', PORT / 'tests/shell_probe.c']
    debug, app = stage / 'shell-probe.com.dbg', stage / 'shell-probe.exe'
    commands = [
        [sdk / 'bin/x86_64-unknown-cosmo-cc', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror',
         '-I' + str(PORT / 'src'), *sources, '-o', debug],
        [sdk / 'bin/ape-x86_64.elf', sdk / 'bin/apelink', '-o', app, '-l', sdk / 'bin/ape-x86_64.elf', debug],
    ]
    try:
        for command in commands:
            subprocess.run(list(map(str, command)), check=True, timeout=120)
        build.set_stack(app)
        target = output / app.name
        os.replace(app, target)
        shutil.copy2(sdk / 'bin/ape-x86_64.elf', output / 'ape-x86_64.elf')
        metadata = {'sdk_sha256': build.SDK_SHA256, 'commands': [list(map(str,c)) for c in commands],
                    'sources': {str(p.relative_to(PORT)): digest(p) for p in sources},
                    'shell_header_sha256': digest(PORT / 'src/shell.h'),
                    'artifact': {'file':target.name, 'bytes':target.stat().st_size, 'sha256':digest(target)}}
        (output / 'BUILD.json').write_text(json.dumps(metadata, indent=2)+'\n')
        return target
    finally:
        shutil.rmtree(stage)


def verify(probe, loader, output):
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Use a fresh or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema':1, 'status':'failed', 'scope':'Production C shell, mocked application dispatcher; no HTTP/model/GPU',
              'platform':platform.platform(), 'artifact_sha256_before':digest(probe), 'cases':[]}
    started = time.monotonic()
    command = ([str(loader)] if loader else []) + [str(probe)]
    try:
        with tempfile.TemporaryDirectory(prefix='cosmo-shell-') as temporary:
            work = Path(temporary)
            def run(label, script=None, *, file=None, stdin=None, code=0):
                args = ['-c',script] if script is not None else ['--file',str(file)] if file else []
                env = {'PATH':'/no-tools','LANG':'C','LC_ALL':'C','TMPDIR':str(work)}
                if os.name == 'nt':
                    for key in ('SystemRoot','WINDIR'):
                        if key in os.environ: env[key] = os.environ[key]
                result = subprocess.run(command + args, input=stdin, cwd=work, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                out, err = output/(label+'.stdout.log'), output/(label+'.stderr.log')
                out.write_bytes(result.stdout); err.write_bytes(result.stderr)
                expected = code << 8 if os.name == 'nt' else code
                record = {'name':label,'arguments':args,'stdin_sha256':hashlib.sha256(stdin).hexdigest() if stdin else None,
                          'returncode':result.returncode,'expected_application_exit':code,'expected_host_exit':expected,
                          'stdout':{'path':out.name,'bytes':out.stat().st_size,'sha256':digest(out)},
                          'stderr':{'path':err.name,'bytes':err.stat().st_size,'sha256':digest(err)}}
                report['cases'].append(record)
                require(result.returncode == expected, f'{label}: exit {result.returncode}, expected {expected}: {result.stderr!r}')
                require(b'process_cwd_unchanged=1\n' in result.stderr, label+': process cwd changed')
                return result.stdout
            def captures(data):
                return [json.loads(line) for line in data.decode().splitlines() if line.startswith('{')]

            tokens = captures(run('literal-arguments', r'''capture "" 'two words' "C:\new\folder\model.gguf" 'a;&&||#b' '$HOME' '*.gguf' '--device' # comment
capture next'''))
            require(tokens[0]['argv'] == ['capture','','two words',r'C:\new\folder\model.gguf','a;&&||#b','$HOME','*.gguf','--device'], 'Quoting changed literal arguments')
            require(tokens[1]['argv'] == ['capture','next'], 'Comment consumed following command')
            tokens = captures(run('parsed-argv-ownership', 'capture-mutate-input first; capture "still owned"'))
            require([x['argv'] for x in tokens] == [['capture-mutate-input','first'],['capture','still owned']], 'Parsed argv aliases modified input')
            run('directory-state', 'mkdir -p "space dir/nested"; cd "space dir"; capture location; cd nested; pwd')
            records = captures((output/'directory-state.stdout.log').read_bytes())
            require(os.path.samefile(host_path(records[0]['process_directory']), work), 'cd changed the actual process cwd')
            require(os.path.samefile(host_path(records[0]['directory']), work/'space dir'), 'cd selected a different actual shell directory')
            source = work/'binary.dat'; source.write_bytes(bytes(range(256))*513+b'\0last\xff')
            before = digest(source)
            data = run('copy-move-binary', 'cp binary.dat "space dir"; mv "space dir/binary.dat" "space dir/copied.dat"; test -f "space dir/copied.dat"; cat "space dir/copied.dat"')
            require(data == source.read_bytes() and digest(work/'space dir/copied.dat') == before, 'cp/mv/cat changed binary bytes')
            run('same-file-refusal', 'cp binary.dat binary.dat || capture rejected')
            require(digest(source) == before, 'Same-file cp damaged input')
            target=work/'keep.dat'; target.write_bytes(b'keep original')
            run('failed-copy-preserves-output', 'cp missing.dat keep.dat || capture rejected')
            require(target.read_bytes()==b'keep original', 'Failed cp destroyed destination')
            values=captures(run('chain-semantics', 'false && capture skipped || capture recovered; true || capture skipped && capture final; capture-status 1 || capture handled'))
            require([v['argv'] for v in values] == [['capture','recovered'],['capture','final'],['capture-status','1'],['capture','handled']], 'Conditional chain semantics differ')
            run('unhandled-chain-fails-fast','false && capture skipped; touch should-not-exist',code=1)
            require(not (work/'should-not-exist').exists(), 'Unhandled failure did not stop script')
            for label, suffix in [('unterminated-quote','capture "bad'),('bare-pipe','capture x | capture y'),('bare-background','capture x &'),('redirection','capture x > output'),('missing-operand','true &&')]:
                run('syntax-'+label,'touch forbidden-'+label+'; '+suffix,code=2)
                require(not (work/('forbidden-'+label)).exists(), 'Syntax was not validated before side effects')
            script=work/'source with spaces.ed'; script.write_bytes(b'capture sourced\r\ntrue\r\n')
            values=captures(run('source-and-file','source "source with spaces.ed"; capture after'))
            require([v['argv'] for v in values]==[['capture','sourced'],['capture','after']], 'source command ordering failed')
            run('file-crlf',file=script)
            script.write_bytes(b'touch nul-file-prefix\0ignored\n')
            run('file-nul-rejected',file=script,code=2)
            require(not (work/'nul-file-prefix').exists(), 'NUL file executed prefix')
            run('stdin-nul-rejected',stdin=b'touch nul-stdin-prefix\0ignored\n',code=2)
            require(not (work/'nul-stdin-prefix').exists(), 'NUL stdin executed prefix')
            run('stdin-oversized-line',stdin=b'touch too-long '+b'x'*(1024*1024)+b'\n',code=2)
            require(not (work/'too-long').exists(), 'Oversized stdin executed prefix')
            run('no-host-process','sh -c "touch external-process"',code=127)
            require(not (work/'external-process').exists(), 'Unknown command launched host shell')
            run('explicit-exit','exit 7; touch after-exit',code=7)
            require(not (work/'after-exit').exists(), 'exit did not stop execution')
            run('remove-files','rm "space dir/copied.dat"; rm -f nonexistent; rmdir "space dir/nested"; rmdir "space dir"')
            require(not (work/'space dir').exists() and not list(work.rglob('*.tmp.*')), 'File operations leaked temporary files')
            report['status']='passed'
    except Exception as error:
        report['error']=str(error)
    finally:
        report['artifact_sha256_after']=digest(probe)
        if report['artifact_sha256_after'] != report['artifact_sha256_before']:
            report.update(status='failed',error='Probe executable changed')
        report['elapsed_seconds']=round(time.monotonic()-started,6)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',action='store_true');p.add_argument('--sdk',type=Path,default=PORT/'out/sdk')
    p.add_argument('--out',type=Path,default=PORT/'out/shell-test');p.add_argument('--probe',type=Path)
    p.add_argument('--loader',type=Path);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();out=a.out.resolve()
    probe=build_probe(a.sdk.resolve(),out) if a.build else (a.probe or out/'shell-probe.exe').resolve()
    loader=None if os.name=='nt' else (a.loader or probe.parent/'ape-x86_64.elf').resolve()
    report=verify(probe,loader,(a.output_dir or out/'results').resolve())
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),'error':report.get('error')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
