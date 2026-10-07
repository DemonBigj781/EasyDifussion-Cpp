#!/usr/bin/env python3
"""Model-free real-application HTTP configuration and restart smoke test.

Uses the produced APE/PE, native Crow routes, and actual persisted JSON. The
embedded provider is selected for deterministic setup. This does not execute
models, test physical hardware, assert busy-generation behavior, or run a browser.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time

from verify_inference_api import Client
from verify_runtime import APPLICATION, LOADER, capture_logs, runtime_environment, sha256, verify_package

PREFIX = '/v1/sdapi/v1/'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unused_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--expected-sha256', default=os.environ.get('EXPECTED_SHA256'))
    parser.add_argument('--logs', '--output-dir', dest='logs', type=Path, default=Path('configuration-api-results'))
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    artifact, logs = args.artifact_dir.resolve(), args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'status': 'FAIL', 'platform': platform.platform(),
              'scope': 'real native configuration HTTP and restart; no models, busy generation, browser, or hardware',
              'requests': [], 'servers': []}
    deadline = started + args.timeout
    executable = artifact / APPLICATION
    before = None
    try:
        before = verify_package(artifact, args.expected_sha256)
        report['application_sha256_before'] = before
        manifest = json.loads((artifact / 'BUILD.json').read_text())
        command = [str(executable)] if os.name == 'nt' else [str(artifact / LOADER), str(executable)]
        with tempfile.TemporaryDirectory(prefix='cosmo-config-api-') as temporary:
            work = Path(temporary)
            initial_models = work / 'first-checkpoints'
            initial_models.mkdir()
            (work / 'second-checkpoints').mkdir()
            config = work / 'easy-diffusion.json'
            require(not config.exists(), 'Smoke test must begin without a configuration file')
            first_port, second_port = unused_port(), unused_port()
            while second_port == first_port:
                second_port = unused_port()

            @contextmanager
            def server(label, port, arguments):
                observation = {'name': label, 'arguments': arguments, 'port': port}
                report['servers'].append(observation)
                process = None
                with capture_logs(logs / (label + '.log')) as (output, errors):
                    process = subprocess.Popen(command + arguments, cwd=work,
                                               env=runtime_environment(work), stdin=subprocess.DEVNULL,
                                               stdout=output, stderr=errors)
                    client = Client(f'http://127.0.0.1:{port}', deadline, report['requests'])
                    try:
                        ready_deadline = min(deadline, time.monotonic() + 18)
                        while time.monotonic() < ready_deadline:
                            require(process.poll() is None, f'{label} exited before readiness: {process.returncode}')
                            try:
                                client.expect(PREFIX + 'config', timeout=0.5)
                                break
                            except (OSError, TimeoutError):
                                time.sleep(0.05)
                        else:
                            raise TimeoutError(f'{label} did not start before readiness deadline')
                        yield client
                    finally:
                        observation['status_before_cleanup'] = process.poll()
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=min(10, max(1, deadline - time.monotonic())))
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                                observation['forced_kill'] = True
                        observation['exit_status'] = process.returncode
                        observation['cleanup_method'] = 'TerminateProcess' if os.name == 'nt' else 'SIGTERM'
                        if config.exists():
                            shutil.copy2(config, logs / (label + '-configuration.json'))
                require(observation['status_before_cleanup'] is None, f'{label} exited unexpectedly before cleanup')
                require(not observation.get('forced_kill'), f'{label} required forced kill')
                expected = 1 if os.name == 'nt' else 0
                require(observation['exit_status'] == expected, f'{label} cleanup status {observation["exit_status"]}; expected {expected}')

            with server('first', first_port, ['sdkit', '--backend', 'cpu', '--provider', 'embedded',
                                             '--port', str(first_port), '--ckpt-dir', str(initial_models)]) as client:
                require(config.is_file(), 'First ordinary server run did not persist default configuration')
                initial = json.loads(config.read_text())
                require(initial['compute'] == {'backend': 'cpu', 'provider': 'auto', 'device': 'auto'}, 'First-run compute defaults differ')
                require(initial['server']['port'] == 8188, 'CLI port override was silently persisted')
                require(initial['models']['checkpoint_dir'] == 'models/checkpoints', 'CLI model directory was silently persisted')
                first = client.expect('/get/app_config')
                require(first == client.expect(PREFIX + 'config'), 'Configuration GET aliases disagree')
                require(first['saved'] == initial, 'GET saved settings differ from persisted JSON')
                require(first['effective']['compute'] == {'backend': 'cpu', 'provider': 'embedded', 'device': 'auto'}, 'CLI provider override missing from effective configuration')
                require(first['effective']['server']['port'] == first_port, 'Server did not use effective CLI port')
                report['initial'] = first
                capabilities = client.expect(PREFIX + 'cosmopolitan-capabilities')
                require(capabilities.get('persistent_configuration') is True and capabilities.get('configuration_schema') == 1,
                        'Native capability response does not expose configuration support')
                pages = []
                for page in ('/cpp-ui/settings', '/cpp-ui/settings/gpu'):
                    status, html, headers = client.request(page, raw=True)
                    scripts = re.findall(rb'<script\s+[^>]*src="([^"]+)"', html)
                    require(status == 200 and 'text/html' in headers.get('Content-Type', ''), f'{page} is unavailable')
                    require(scripts == [b'/cpp-ui/scripts/kiosk.js', b'/cpp-ui/scripts/backend-platform.js'], f'{page} still loads unsupported legacy scripts')
                    require(b'id="backend_platform"' in html and b'id="save-system-settings-btn"' in html, f'{page} lacks existing Settings controls')
                    pages.append({'path': page, 'bytes': len(html), 'sha256': hashlib.sha256(html).hexdigest()})
                script_path = '/cpp-ui/scripts/backend-platform.js'
                status, script, headers = client.request(script_path, raw=True)
                require(status == 200 and 'javascript' in headers.get('Content-Type', ''), 'Portable Settings script unavailable')
                digest = hashlib.sha256(script).hexdigest()
                stored = manifest['embedded_resources'][script_path.lstrip('/')]
                require(digest == stored['sha256'] and len(script) == stored['bytes'], 'Served Settings script differs from packaged resource')
                for marker in (b'/get/app_config', b'/app_config', b'restart_required', b'/v1/sdapi/v1/backend-devices'):
                    require(marker in script, f'Portable Settings script lacks native API contract: {marker}')
                report['ui'] = {'pages': pages, 'script': {'path': script_path, 'bytes': len(script), 'sha256': digest}, 'browser_executed': False}
                update = {'compute': {'backend': 'cpu', 'provider': 'embedded', 'device': 'CPU'},
                          'server': {'port': second_port, 'log_level': 'warning'},
                          'models': {'checkpoint_dir': 'second-checkpoints'}}
                changed = client.expect('/app_config', update)
                saved = json.loads(config.read_text())
                require(changed['saved'] == saved and all(saved[k] == v for k, v in update.items()), 'POST config choices were not saved exactly')
                require(changed['effective'] == first['effective'] and changed['restart_required'] is True, 'Startup mutation changed running settings or hid restart requirement')
                require(changed == client.expect('/get/app_config'), 'Saved config GET did not reflect mutation')
                options = client.expect(PREFIX + 'options')
                client.expect(PREFIX + 'options', {'CLIP_stop_at_last_layers': 2})
                options['CLIP_stop_at_last_layers'] = 2
                require(client.expect(PREFIX + 'options') == options, 'Live native options did not change')
                require(json.loads(config.read_text())['options'] == options, 'Live options were not persisted to the same config file')
                live = client.expect(PREFIX + 'config')
                require(live['effective']['options'] == options and live['saved']['options'] == options, 'Config API disagrees with live options')
                stable_hash = sha256(config)
                for endpoint, invalid in [('/app_config', {'unknown': True}),
                                          (PREFIX + 'config', {'compute': {'backend': 'cuda'}}),
                                          (PREFIX + 'config', {'server': {'port': 0}}),
                                          (PREFIX + 'options', {'live_previews_enable': True})]:
                    client.expect(endpoint, invalid, status=400)
                    require(sha256(config) == stable_hash, 'Rejected settings mutation changed disk contents')
                require(client.expect(PREFIX + 'config') == live, 'Rejected mutation changed running or saved settings')
                report['saved_before_restart'] = live
            with server('restarted', second_port, ['sdkit', '--config', str(config)]) as client:
                current = client.expect(PREFIX + 'config')
                require(current['saved'] == json.loads(config.read_text()), 'Restart GET differs from saved file')
                require(current['effective']['compute'] == update['compute'], 'Restart ignored saved provider/device/backend')
                require(current['effective']['server'] == update['server'], 'Restart ignored saved server settings')
                require(current['effective']['models']['checkpoint_dir'].replace('\\', '/').endswith('/second-checkpoints'), 'Restart did not resolve saved relative model directory')
                require(current['restart_required'] is False, 'Identical saved/running settings still require restart')
                require(client.expect(PREFIX + 'options') == options, 'Restart lost persisted generation options')
                require(current == client.expect('/get/app_config'), 'GET aliases disagree after restart')
                report['after_restart'] = current
            require(not list(work.glob('*.tmp.*')), 'Configuration temporary files leaked')
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = str(error)
    finally:
        if executable.exists():
            report['application_sha256_after'] = sha256(executable)
            if before is not None and report['application_sha256_after'] != before:
                report['status'] = 'FAIL'
                report['error'] = 'Application changed during configuration verification'
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report['evidence'] = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                              for p in sorted(logs.iterdir()) if p.is_file() and p.name != 'report.json']
        (logs / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds'],
                      'report': str(logs / 'report.json'), 'error': report.get('error')}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
