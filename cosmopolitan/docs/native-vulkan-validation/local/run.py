"""One exact-workload positive reproduction; not a reusable product verifier."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ARTIFACT = ROOT / 'cosmopolitan/out/native-final-ci/application'
sys.path.insert(0, str(ARTIFACT))
from verify_inference import inspect_png, unique_marker, COUNTERS
from verify_runtime import sha256, runtime_environment

old_path = ROOT / 'cosmopolitan/out/native-loader-crash-repro/direct-report.json'
old = json.loads(old_path.read_text())
command = old['command'].copy()
command[0] = str(ARTIFACT / 'ape-x86_64.elf')
command[1] = str(ARTIFACT / 'easy-diffusion.exe')
command[command.index('--output') + 1] = str(OUT / 'image.png')
app, model = Path(command[1]), Path(command[command.index('--model') + 1])
tmp = OUT / 'tmp'
tmp.mkdir(exist_ok=False)
env = runtime_environment(tmp)
env.update(PATH='/usr/bin:/bin', LP_NUM_THREADS='2',
           VK_DRIVER_FILES=old['vulkan_icd'], VK_ICD_FILENAMES=old['vulkan_icd'])
expected_app = '7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9'
report = {'schema': 1, 'success': False,
          'scope': 'One exact previously failing direct native image command with corrected CI application; 2-step plumbing, not image quality or physical GPU validation',
          'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'prior_report': {'path': str(old_path), 'sha256': sha256(old_path), 'returncode': old['returncode'], 'app_sha256': old['app_sha256']},
          'source': json.loads((ARTIFACT / 'BUILD.json').read_text())['source'],
          'command': command, 'cwd': str(OUT), 'environment': env, 'timeout_seconds': 1800,
          'application': {'path': str(app), 'bytes': app.stat().st_size, 'sha256_before': sha256(app)},
          'model': {'path': str(model), 'bytes': model.stat().st_size, 'sha256_before': sha256(model)},
          'software_driver_threads': 2, 'traced': False,
          'verification_helpers': {name: sha256(ARTIFACT / name) for name in ('verify_inference.py', 'verify_runtime.py')},
          'evidence': {}}

def save():
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

def check_output():
    stdout = (OUT / 'direct.stdout.log').read_text(errors='replace')
    stderr = (OUT / 'direct.stderr.log').read_text(errors='replace')
    lines = stdout.splitlines()
    png = inspect_png(OUT / 'image.png', 256, 256)
    unique_marker(lines, 'IMAGE_INFERENCE', r'IMAGE_INFERENCE PASS')
    stages = {}
    for stage in ('MODEL_LOAD', 'GENERATION'):
        m = unique_marker(lines, 'IMAGE_' + stage,
            r'IMAGE_' + stage + r' backend=webgpu success=1 seconds=([0-9.]+) graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+)')
        seconds = float(m.group(1))
        assert math.isfinite(seconds) and seconds >= 0, stage
        stages[stage.lower()] = {'seconds': seconds, **dict(zip(COUNTERS, map(int, m.groups()[1:])))}
    assert all(stages['generation'][k] > 0 for k in COUNTERS)
    params = unique_marker(lines, 'IMAGE_PARAMETERS', r'IMAGE_PARAMETERS width=256 height=256 steps=2 seed=42 threads=2 cfg_scale=7 sampler=(\S+) scheduler=(\S+) cpu_fallback_allowed=1')
    m = unique_marker(lines, 'IMAGE_SAMPLING', r'IMAGE_SAMPLING backend=webgpu first_step=1 last_step=2 total_steps=2 completed_callbacks=2 intervals=1 monotonic=1 graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+)')
    interval = dict(zip(COUNTERS, map(int, m.groups())))
    assert all(0 < interval[k] <= stages['generation'][k] for k in COUNTERS)
    assert interval['matmuls'] <= interval['dispatches'] and stages['generation']['matmuls'] <= stages['generation']['dispatches']
    progress = []
    for line in stderr.splitlines():
        if not line.startswith('IMAGE_SAMPLE_PROGRESS'):
            continue
        m = re.fullmatch(r'IMAGE_SAMPLE_PROGRESS step=(\d+) total_steps=2 seconds=([0-9.]+)', line)
        assert m is not None, line
        seconds = float(m.group(2))
        assert math.isfinite(seconds) and seconds >= 0
        progress.append({'step': int(m.group(1)), 'seconds': seconds})
    assert [p['step'] for p in progress] == [1, 2], progress
    execution = unique_marker(lines, 'IMAGE_WEBGPU_EXECUTION', r'IMAGE_WEBGPU_EXECUTION provider=native software=1 native_loader_opens=(\d+) cpu_fallback_allowed=1 cpu_fallback_measured=0')
    assert int(execution.group(1)) > 0
    adapter = unique_marker(lines, 'IMAGE_WEBGPU_ADAPTER', r'IMAGE_WEBGPU_ADAPTER (.+)').group(1)
    assert 'llvmpipe' in adapter.lower(), adapter
    output = unique_marker(lines, 'IMAGE_OUTPUT', r'IMAGE_OUTPUT width=(\d+) height=(\d+) channels=(\d+) pixel_bytes=(\d+) png_bytes=(\d+) pixel_min=(\d+) pixel_max=(\d+) pixel_mean=([0-9.]+)')
    assert tuple(map(int, output.groups()[:7])) == tuple(png[k] for k in ('width','height','channels','pixel_bytes','bytes','pixel_min','pixel_max'))
    assert math.isclose(float(output.group(8)), png['pixel_mean'], abs_tol=1e-7)
    report.update(png=png, execution={**stages, 'sampler': params.group(1), 'scheduler': params.group(2),
        'generation_scope': 'Complete generate_image after model load; CPU fallback allowed, fraction not measured',
        'sampling': {'first_step': 1, 'last_step': 2, 'total_steps': 2, 'completed_callbacks': 2, 'intervals': 1,
                     'monotonic': True, **interval, 'progress': progress,
                     'scope': 'Between completed denoising steps, excludes first step, CLIP and VAE; previews disabled'},
        'provider': 'native', 'software_adapter': True, 'adapter': adapter, 'native_loader_opens': int(execution.group(1)),
        'hardware_verified': False, 'cpu_fallback_allowed': True, 'cpu_fallback_fraction_measured': False})

save()
try:
    assert report['application']['sha256_before'] == expected_app and app.stat().st_size == 190491826
    assert report['model']['sha256_before'] == old['model_sha256']
    assert report['source'] == {'commit': '1730424270e35ff4a06bd978654c9990cfd5777e', 'dirty': False, 'repository': 'DemonBigj781/EasyDifussion-Cpp'}
    started = time.monotonic()
    print('STARTED exact native image reproduction', flush=True)
    with (OUT / 'direct.stdout.log').open('wb') as stdout, (OUT / 'direct.stderr.log').open('wb') as stderr:
        proc = subprocess.Popen(command, cwd=OUT, env=env, stdout=stdout, stderr=stderr)
        report['launcher_pid'] = proc.pid
        save()
        try:
            report['returncode'] = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            report['returncode'] = proc.returncode
            report['timed_out'] = True
            raise RuntimeError('Application exceeded the 1800-second deadline')
        finally:
            report['seconds'] = round(time.monotonic() - started, 6)
    assert report['returncode'] == 0, f"Application exit {report['returncode']}"
    check_output()
    report['success'] = True
except Exception as error:
    report['error'] = f'{type(error).__name__}: {error}'
finally:
    for key, path in (('application', app), ('model', model)):
        report[key]['sha256_after'] = sha256(path)
        report[key]['unchanged'] = report[key]['sha256_before'] == report[key]['sha256_after']
        if not report[key]['unchanged']:
            report['success'] = False
            report['error'] = key + ' hash changed'
    for name in ('direct.stdout.log', 'direct.stderr.log', 'image.png', 'run.py'):
        p = OUT / name
        if p.exists():
            report['evidence'][name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    save()
    print(json.dumps({'success': report['success'], 'returncode': report.get('returncode'), 'seconds': report.get('seconds'), 'error': report.get('error'), 'report': str(OUT / 'report.json')}, indent=2), flush=True)
sys.exit(0 if report['success'] else 1)
