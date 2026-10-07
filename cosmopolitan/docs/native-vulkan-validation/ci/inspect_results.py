#!/usr/bin/env python3
"""Offline inspection of retained native-provider CI evidence. Never launches a child or uses network.

Use the matching checkout's tests directory as --parser-dir. Only its pure hash,
PNG, and log parsers are called; packaged parser bytes must match first. Success
means retained evidence is consistent, not independent rerunning of a CI job.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from types import SimpleNamespace
import zipfile

COUNTERS = ('graphs', 'submissions', 'dispatches', 'matmuls', 'readbacks')
POLICY_CASES = ('hardware-first', 'embedded-fallback', 'native-requires-hardware',
                'explicit-native-software', 'missing-and-ambiguous', 'configured-default',
                'immutable-factory', 'native-thread-guard')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class Inspector:
    def __init__(self, args):
        self.args, self.inventory, self.groups = args, {}, []
        self.bound_evidence = {}
        self.roots = {key: getattr(args, key.replace('-', '_')).resolve()
                      for key in ('app', 'linux', 'windows', 'linux-native', 'diagnostics')}

    def file(self, group, name):
        rel = PurePosixPath(name)
        need(name and not rel.is_absolute() and not PureWindowsPath(name).is_absolute()
             and '\\' not in name and '..' not in rel.parts, 'unsafe evidence path: ' + name)
        root = self.roots[group]
        path = root.joinpath(*rel.parts)
        need(path.is_file() and path.resolve().is_relative_to(root), f'missing/escaped file {group}/{name}')
        key = group + '/' + name
        for prefix, names in self.bound_evidence.items():
            if key.startswith(prefix) and not key.endswith('/report.json'):
                need(key in names, 'raw file absent from report hash inventory: ' + key)
        if key not in self.inventory:
            self.inventory[key] = {'path': key, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        return path

    def read(self, group, name):
        return self.file(group, name).read_text(errors='replace')

    def json(self, group, name):
        return json.loads(self.read(group, name))

    def report(self, group, subdir, status='passed', identity=True):
        path = (subdir + '/' if subdir else '') + 'report.json'
        value = self.json(group, path)
        need(value.get('status') == status and not value.get('error'), f'{group}/{path} did not pass')
        if group in ('linux', 'linux-native', 'windows'):
            need(value.get('platform', '').startswith('Windows' if group == 'windows' else 'Linux'), 'report platform differs')
        if identity:
            self.identity(value)
        if 'evidence' in value:
            evidence = value['evidence']
            if isinstance(evidence, dict):
                evidence = [dict(record, path=name) for name, record in evidence.items()]
            need(isinstance(evidence, list) and evidence, 'empty evidence inventory: ' + path)
            names = set()
            for record in evidence:
                name = record['path']
                need(name not in names, 'duplicate evidence record')
                names.add(name)
                actual = self.file(group, subdir + '/' + name)
                need(record['bytes'] == actual.stat().st_size and record['sha256'] == digest(actual),
                     f'changed evidence bytes: {group}/{subdir}/{name}')
            prefix = group + '/' + subdir + '/'
            self.bound_evidence[prefix] = {prefix + name for name in names}
        return value

    def identity(self, value):
        before = value.get('sha256_before', value.get('application_sha256_before'))
        after = value.get('sha256_after', value.get('application_sha256_after'))
        need(before == after == self.app_sha, 'report application identity differs')

    def record(self, name, function, *args):
        try:
            detail = function(*args)
            self.groups.append({'group': name, 'status': 'passed', 'details': detail})
        except Exception as error:
            self.groups.append({'group': name, 'status': 'failed', 'error': str(error)})

    def package(self):
        manifest = self.json('app', 'BUILD.json')
        need(manifest['source']['commit'] == self.args.expected_source, 'wrong source commit')
        need(manifest['source']['dirty'] is False, 'artifact was built from dirty source')
        app = self.file('app', 'easy-diffusion.exe')
        self.app_sha = digest(app)
        if self.args.expected_sha256:
            need(self.app_sha == self.args.expected_sha256, 'wrong expected executable SHA')
        need(manifest['executable']['sha256'] == self.app_sha and
             manifest['executable']['bytes'] == app.stat().st_size, 'BUILD executable metadata differs')
        with zipfile.ZipFile(app) as archive:
            embedded = json.loads(archive.read('BUILD.json'))
        need(embedded == {k: v for k, v in manifest.items() if k != 'executable'}, 'embedded BUILD differs')
        modules = ('verify_runtime', 'verify_bootstrap', 'verify_inference', 'verify_inference_api',
                   'verify_webgpu_device', 'verify_configuration_api', 'verify_windows_native_discovery',
                   'verify_config', 'verify_device_policy')
        for module in modules:
            shipped = self.file('app', module + '.py')
            source = self.args.parser_dir / (module + '.py')
            need(source.is_file() and digest(shipped) == digest(source), 'parser checkout mismatch: ' + module)
        # Imports define host verifiers but invoke no main(), subprocess, or network function.
        sys.path.insert(0, str(self.args.parser_dir.resolve()))
        self.runtime = importlib.import_module('verify_runtime')
        self.bootstrap = importlib.import_module('verify_bootstrap')
        self.image = importlib.import_module('verify_inference')
        self.device = importlib.import_module('verify_webgpu_device')
        need(self.runtime.verify_package(self.roots['app'], self.app_sha) == self.app_sha, 'package validation failed')
        self.manifest = manifest
        self.pin = self.json('app', 'PIN.json')
        symbols = self.json('app', 'SYMBOLS.json')
        need(symbols.get('status') == 'passed' and symbols['legacy_versioned_ggml_symbols'] == 0 and
             all(count == 1 for count in symbols['required_symbol_counts'].values()) and
             symbols['shared_ggml_webgpu_sources'] == ['ggml-webgpu/ggml-webgpu.cpp'], 'shared symbol audit did not pass')
        link = self.file('app', 'software-webgpu/LINK.json')
        need(digest(link) == manifest['software_webgpu_metadata_sha256'] and
             json.loads(link.read_text()) == manifest['software_webgpu'], 'static dependency provenance differs')
        loader = self.file('app', 'ape-x86_64.elf')
        need(loader.open('rb').read(7) == b'\x7fELF\x02\x01\x01', 'external loader is not x86-64 ELF')
        for filename, metadata, field in [('config-probe.exe', 'CONFIG_PROBE_BUILD.json', 'artifact'),
                                           ('device-policy-test.exe', 'DEVICE_POLICY_BUILD.json', 'artifact_sha256')]:
            probe = self.file('app', filename)
            info = self.json('app', metadata)
            recorded = info[field]['sha256'] if isinstance(info[field], dict) else info[field]
            need(digest(probe) == recorded, 'packaged probe identity differs: ' + filename)
        return {'source': manifest['source'], 'app_sha256': self.app_sha,
                'embedded_resources': len(manifest['embedded_resources']), 'symbols': 'passed'}

    def model(self, report):
        item = report['model']
        for field in ('expected_sha256', 'sha256_before', 'sha256_after'):
            need(item[field] == self.pin['sha256'], 'wrong/changed model digest')
        need(item['bytes'] == self.pin['bytes'] and report['model_hash_unchanged'] is True,
             'wrong/changed model size')
        need(report['model_pin']['metadata'] == self.pin, 'model pin metadata differs')
        need(report['model_pin']['sha256'] == digest(self.file('app', 'PIN.json')), 'model pin bytes differ')

    def runtime_gate(self, group, subdir, isolated):
        r = self.report(group, subdir)
        need(r['isolated'] is isolated, 'incorrect isolation scope')
        windows = group == 'windows'
        need(set(r['initial_root_entries']) == ({'easy-diffusion.exe', 'tmp'} if windows else
             {'easy-diffusion.exe', 'ape-x86_64.elf', 'tmp'}), 'unexpected runtime root entries')
        tests = {t['name']: t for t in r['tests']}
        expected = {'self-test': 0, 'devices': 0, 'diffusion-command': 0, 'image-command': 0,
                    'training-command': 0, 'training-unmatched-option': 2, 'image-unmatched-option': 2,
                    'image-invalid-dimensions': 2, 'image-invalid-backend': 2}
        need(set(tests) == set(expected) | {'native-server-and-embedded-ui'}, 'missing/duplicate runtime commands')
        need(len(tests) == len(r['tests']), 'duplicate runtime test')
        for label, code in expected.items():
            test = tests[label]
            host = code << 8 if windows else code
            need((test['exit_code'], test['expected_host_exit_status'], test['expected_application_exit_code']) ==
                 (host, host, code), 'wrong native/logical return code: ' + label)
            stdout = self.read(group, subdir + '/' + test['stdout_log'])
            stderr = self.read(group, subdir + '/' + test['stderr_log'])
            self.file(group, subdir + '/' + test['log'])
            text = stdout + '\n' + stderr
            need(text.strip(), 'empty command output')
            if label == 'self-test':
                need(self.bootstrap.validate_selftest_output(text) == test['greedy_token_ids'], 'token evidence differs')
                need(self.runtime.validate_webgpu_output(text) == test['webgpu'], 'WebGPU evidence differs')
                need(self.runtime.validate_inplace_output(text) == test['inplace_storage'], 'alias evidence differs')
            markers = {'training-unmatched-option': 'Missing value for native trainer option',
                       'image-unmatched-option': 'unknown, duplicate, or incomplete option',
                       'image-invalid-dimensions': 'invalid value for --width',
                       'image-invalid-backend': 'compute.backend must be cpu or webgpu'}
            if label in markers:
                need(markers[label] in text, 'missing deliberate rejection diagnostic')
        server = tests['native-server-and-embedded-ui']
        need(server['ping_status'] == 200 and server['parent_traversal_status'] in (400, 404), 'HTTP security/ping failed')
        need(server['checks'] and all(t['status'] == 200 and t['bytes'] > 0 for t in server['checks']), 'UI checks failed')
        for key in ('log', 'stdout_log', 'stderr_log'):
            self.file(group, subdir + '/' + server[key])
        return {'isolated': isolated, 'native_windows_negative_status': 512 if windows else None,
                'command_count': len(tests), 'shared_alias_checks': 12}

    def bootstrap_gate(self):
        group, subdir = 'linux', 'results-linux-bootstrap'
        r = self.report(group, subdir)
        need(r['filesystem_isolated'] is False and r['explicit_external_loader_supplied'] is False and
             r['preinstalled_ape_on_path'] is False and r['initial_runtime_directory_empty'] is True and
             r['temporary_directory_removed'] is True and r['exit_code'] == 0 and
             r['sha256_copy_after'] == self.app_sha and r['hash_unchanged'] is True, 'bootstrap scope/identity failure')
        need(set(r['path_utilities']) == {'uname', 'mkdir', 'dd', 'gzip', 'chmod', 'mv'}, 'bootstrap utilities differ')
        text = self.read(group, subdir + '/' + r['log'])
        need(self.bootstrap.validate_selftest_output(text) == r['greedy_token_ids'] and
             self.runtime.validate_webgpu_output(text) == r['webgpu'], 'bootstrap log/report mismatch')
        return {'scope': 'host /bin/sh and utilities, transient embedded loader extraction; not chroot'}

    def image_gate(self, group, backend):
        subdir = 'results-inference-' + backend
        r = self.report(group, subdir)
        self.model(r)
        need(r['backend'] == backend and r['exit_code'] == 0 and r['app_hash_unchanged'] is True, 'image execution failed')
        need(r['provider_policy'] == 'embedded' and r['filesystem_isolated'] is False, 'image provider/scope differs')
        need(r['launch_mode'] == ('native Windows PE' if group == 'windows' else 'explicit APE ELF loader'), 'wrong image launch mode')
        params = r['parameters']
        need((params['width'], params['height'], params['steps']) == (256, 256, 4 if backend == 'cpu' else 2), 'wrong image gate dimensions/steps')
        png = self.image.inspect_png(self.file(group, subdir + '/' + r['output_png']), params['width'], params['height'])
        need(png == r['png'], 'PNG independent decode differs')
        args = SimpleNamespace(**params, backend=backend, sampler=r['execution']['sampler'], scheduler=r['execution']['scheduler'])
        execution = self.image.verify_markers(self.read(group, subdir + '/inference.stdout.log'),
                        self.read(group, subdir + '/inference.stderr.log'), args, png)
        need(execution == r['execution'], 'image counters/report differ')
        return {'seconds': r['elapsed_seconds'], 'png_sha256': png['sha256'], 'sampling': execution['sampling'],
                'cpu_fallback_fraction_measured': False, 'image_quality_assessed': False}

    def api_gate(self, group, native=False):
        subdir = 'results-native-inference-api' if native else 'results-inference-api'
        r = self.report(group, subdir)
        self.model(r)
        need(r['backend'] == ('webgpu' if native else 'cpu') and r['provider_policy'] == ('native' if native else 'embedded'), 'API provider/backend differs')
        need(r['require_hardware'] is False and r['filesystem_isolated'] is False and
             r['browser_execution_tested'] is False and r['app_hash_unchanged'] is True, 'API scope differs')
        need(r['launch_mode'] == ('native Windows PE' if group == 'windows' else 'explicit APE ELF loader'), 'API launch mode differs')
        need(r['server_exit_before_cleanup'] is None, 'API server died before cleanup')
        need(r['server_cleanup_exit_status'] == (1 if group == 'windows' else 0), 'unexpected API cleanup exit status')
        gen = r['generation']
        need(gen['parameters']['steps'] == (2 if native else 4) and gen['final_progress']['completed'] is True, 'image API did not finish')
        need(gen['persistent_options_unchanged'] is True and all(gen[k] == 409 for k in
             ('concurrent_generation_status', 'concurrent_options_status', 'wrong_task_interrupt_status',
              'finished_task_interrupt_status', 'duplicate_task_id_status')), 'request lifecycle isolation failed')
        ui = r['ui']
        need(ui['browser_execution_tested'] is False, 'API incorrectly claims browser execution')
        for script in ui['scripts']:
            resource = self.manifest['embedded_resources'][script['path'].lstrip('/')]
            need(script['bytes'] == resource['bytes'] and script['sha256'] == resource['sha256'], 'API served script identity differs')
        response = gen['active_progress_responsiveness']
        durations = response['elapsed_seconds']
        need(response['requests'] == len(durations) == 32 and response['timeout_seconds'] == 2 and
             all(math.isfinite(x) and 0 <= x < 2 for x in durations) and response['max_seconds'] == max(durations), 'progress responsiveness failed')
        need(len(r['invalid_overrides']) == 3 and all(t['status'] == 400 and t['gate_released'] for t in r['invalid_overrides']), 'invalid override gate failed')
        png = self.image.inspect_png(self.file(group, subdir + '/api-image.png'), 256, 256)
        need(png == gen['png'], 'API PNG differs from independent decode')
        cancel = r['cancellation']
        if native:
            need(cancel == {'status': 'not-run', 'reason': 'explicit --skip-cancel'}, 'native cancellation scope changed')
        else:
            need(cancel['requested_steps'] == 8 and cancel['early_cancellation_observed'] is True and
                 cancel['gate_released'] is True and cancel['interrupt_status'] == 200 and
                 cancel['generation_status'] in (200, 500) and cancel['final_progress']['completed'] is True and
                 cancel['final_progress']['interrupted'] is True and
                 0 < cancel['final_progress']['current_step'] < 8, 'early task-specific cancellation failed')
        stderr = self.read(group, subdir + '/server.stderr.log')
        count, lane = (1, 'original-main') if native else (2, 'worker')
        marker = 'NATIVE_INFERENCE_MAIN' if native else 'NATIVE_INFERENCE_WORKER'
        stacks = re.findall(r'^' + marker + r' stack_bytes=(\d+) guard_bytes=(\d+) stack_source=pthread_getattr_np$', stderr, re.M)
        workers = [{'stack_bytes': int(s), 'guard_bytes': int(g), 'source': 'pthread_getattr_np', 'lane': lane} for s, g in stacks]
        need(len(workers) == count and workers == r['inference_workers'], 'stack marker evidence differs')
        need(all(w['stack_bytes'] >= 8388608 if native else w['stack_bytes'] == 8388608 and w['guard_bytes'] > 0 for w in workers), 'inadequate measured inference stack')
        records = re.findall(r'^NATIVE_INFERENCE_EXECUTION success=([01]) selector=(\S+) provider=(\S+) software=(-?\d+) adapter_type=(\d+) graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+) native_loader_opens=(\d+)$', stderr, re.M)
        observed = [dict(success=t[0] == '1', selector=t[1], provider=t[2], software=int(t[3]), adapter_type=int(t[4]),
                         **dict(zip((*COUNTERS, 'native_loader_opens'), map(int, t[5:])))) for t in records]
        need(len(observed) == count and observed == r['inference_execution'] and observed[0]['success'], 'inference counters differ')
        if native:
            actual = observed[0]
            need(actual['provider'] == 'native' and actual['software'] == 1 and actual['adapter_type'] == 3 and
                 actual['native_loader_opens'] > 0 and all(actual[c] > 0 for c in COUNTERS), 'native software inference not proven')
            need(actual['selector'] == r['selected_device']['selector'] and r['selected_device']['software'] is True,
                 'selected device differs from execution')
        else:
            need(all(t[c] == 0 for t in observed for c in COUNTERS), 'CPU API unexpectedly used GPU')
        need(all('error' not in event or (event['path'] == '/v1/internal/ping' and event['method'] == 'GET')
                 for event in r['http_events']), 'HTTP transport failure outside readiness probe')
        return {'seconds': r['elapsed_seconds'], 'png_sha256': png['sha256'], 'progress_max_seconds': max(durations),
                'execution': observed, 'cancellation_tested': not native, 'cleanup_status': r['server_cleanup_exit_status'],
                'counter_scope': 'whole inference closure; native API counters are not denoising-only'}

    def configuration_gate(self, group):
        subdir = 'results-configuration-api'
        r = self.report(group, subdir, 'PASS')
        first, saved, after = r['initial'], r['saved_before_restart'], r['after_restart']
        need(first['saved']['compute'] == {'backend': 'cpu', 'provider': 'auto', 'device': 'auto'} and
             first['effective']['compute'] == {'backend': 'cpu', 'provider': 'embedded', 'device': 'auto'}, 'defaults/CLI overlay differ')
        need(saved['restart_required'] is True and after['restart_required'] is False, 'restart requirement not observed')
        need(saved['saved'] == after['saved'] == self.json(group, subdir + '/first-configuration.json') ==
             self.json(group, subdir + '/restarted-configuration.json'), 'persisted restart documents differ')
        need(saved['saved']['compute'] == after['effective']['compute'] == {'backend': 'cpu', 'provider': 'embedded', 'device': 'CPU'}, 'restart ignored saved compute settings')
        need(saved['effective']['compute'] == first['effective']['compute'] and saved['effective']['server'] == first['effective']['server'], 'POST changed startup-only running settings')
        need(after['effective']['server'] == after['saved']['server'] and
             after['effective']['models']['checkpoint_dir'].replace('\\', '/').endswith('/second-checkpoints') and
             after['effective']['options'] == after['saved']['options'] == saved['effective']['options'] and
             after['saved']['options']['CLIP_stop_at_last_layers'] == 2, 'restart lost server/model/live options')
        servers = r['servers']
        need(len(servers) == 2 and [s['name'] for s in servers] == ['first', 'restarted'] and
             servers[0]['port'] != servers[1]['port'], 'missing actual two-server restart')
        need(servers[0]['port'] == first['effective']['server']['port'] and servers[1]['port'] == after['effective']['server']['port'], 'server ports differ from effective configuration')
        need(servers[1]['arguments'][:2] == ['sdkit', '--config'], 'second server did not consume saved config')
        for server in servers:
            need(server['status_before_cleanup'] is None and not server.get('forced_kill') and
                 server['exit_status'] == (1 if group == 'windows' else 0) and
                 server['cleanup_method'] == ('TerminateProcess' if group == 'windows' else 'SIGTERM'), 'configuration server cleanup scope differs')
            for suffix in ('.log', '.stdout.log', '.stderr.log'):
                self.file(group, subdir + '/' + server['name'] + suffix)
        events = r['requests']
        need(all('error' not in e or (e['path'] == '/v1/sdapi/v1/config' and e['method'] == 'GET') for e in events) and
             sum(e.get('status') == 400 for e in events) == 4 and
             any(e['path'] == '/app_config' and e['method'] == 'POST' and e.get('status') == 200 for e in events), 'missing HTTP mutation/rejection evidence')
        ui = r['ui']
        script = ui['script']
        resource = self.manifest['embedded_resources'][script['path'].lstrip('/')]
        need(ui['browser_executed'] is False and script['bytes'] == resource['bytes'] and script['sha256'] == resource['sha256'], 'served Settings script identity/scope differs')
        return {'real_http_restart': True, 'browser_executed': False, 'model_execution': False,
                'cleanup_method': servers[0]['cleanup_method']}

    def device_gate(self, dual):
        group, subdir, policy = 'linux-native', 'results-dual-provider' if dual else 'results-native-device', 'auto' if dual else 'native'
        r = self.report(group, subdir)
        need(r['provider_policy'] == policy and r['require_hardware'] is False and r['all_adapters_same_process'] is dual and
             r['filesystem_isolated'] is False and r['execution_lane'] == 'main-thread-service', 'native graph policy/lane differs')
        commands = {x['label']: x for x in r['commands']}
        need(len(commands) == len(r['commands']), 'duplicate command label')
        stdout = self.read(group, subdir + '/' + commands['graph']['stdout_log'])
        need(stdout.splitlines().count('WEBGPU_DEVICE_LANE mode=main-thread-service') == 1, 'missing actual original-main service exercise')
        devices = self.device.validate_all(stdout, policy) if dual else [self.device.validate_graph(stdout, policy, False)]
        need(len(devices) == len(r['devices']), 'reported device count differs')
        if dual:
            need({x['provider'] for x in devices} == {'native', 'embedded'} and r['native_embedded_coexistence_verified'] is True, 'same-process dual-provider compute missing')
        for i, (parsed, stored) in enumerate(zip(devices, r['devices'])):
            need(all(stored.get(k) == v for k, v in parsed.items()), 'graph report differs from raw numerical log')
            need(parsed['software'] is True and parsed['adapter_type'] == 3 and parsed['hardware_verified'] is False,
                 'software CI unexpectedly claims physical GPU')
            llama = commands['llama-' + str(i)]
            proof = self.device.validate_llama(self.read(group, subdir + '/' + llama['stdout_log']),
                                               self.read(group, subdir + '/' + llama['stderr_log']), parsed)
            need(proof == stored['llama'], 'model inference report differs from raw logs')
            rejection = commands['hardware-rejection-' + str(i)]
            out = self.read(group, subdir + '/' + rejection['stdout_log'])
            err = self.read(group, subdir + '/' + rejection['stderr_log'])
            expected = f"WEBGPU_DEVICE_TEST FAIL: hardware required, selected {parsed['selector']} provider={parsed['provider']} type={parsed['device_kind']}"
            need(err.splitlines().count(expected) == 1 and 'WEBGPU_DEVICE_EXECUTION' not in out and
                 'WEBGPU_DEVICE_TEST PASS' not in out and stored['hardware_required_rejection_verified'] is True, 'software adapter was not rejected by hardware gate')
        need(set(commands) == {'graph'} | {f'{prefix}-{i}' for i in range(len(devices)) for prefix in ('llama', 'hardware-rejection')}, 'missing/unexpected native command')
        for label, command in commands.items():
            expected = 1 if label.startswith('hardware-rejection') else 0
            need(command['exit_code'] == command['expected_host_exit_status'] == command['expected_application_exit_code'] == expected, 'native command status differs')
            for key in ('stdout_log', 'stderr_log'):
                self.file(group, subdir + '/' + command[key])
        need(valid_sha(r['vulkan_icd']['sha256']), 'missing runner ICD hash')
        self.file(group, 'native-driver-packages.txt')
        return {'devices': devices, 'dual_provider_graphs_same_process': dual,
                'llama_runs_are_separate_processes': True, 'physical_hardware_verified': False}

    def windows_discovery(self):
        group, subdir = 'windows', 'results-windows-native-discovery'
        r = self.report(group, subdir)
        need(re.fullmatch(r'[0-9]+(?:\.[0-9]+){3}', r['chrome_version']) and r['primary_sources'], 'missing installed Chrome provenance')
        need(r['copied_sha256_after'] == self.app_sha and r['gpu_compute_tested'] is False and
             r['physical_hardware_tested'] is False, 'discovery identity/scope differs')
        reason = 'ShaderF16 is required by the compiled GGML kernels'
        info = r['unavailable_adapter']
        need(info['provider'] == 'native' and 'swiftshader' in info['name'].lower() and info['reason'] == reason, 'wrong unavailable adapter reason')
        installed = r['installed_inputs']
        need({PureWindowsPath(x['path']).name for x in installed} == {'vulkan-1.dll', 'vk_swiftshader.dll', 'vk_swiftshader_icd.json'} and len(installed) == 3, 'wrong installed DLL inputs')
        for item in installed:
            need(item['bytes'] > 0 and valid_sha(item['sha256_before']) and item['sha256_before'] == item['sha256_after'], 'installed DLL changed')
        need(PureWindowsPath(r['effective_icd']['ICD']['library_path']) == next(PureWindowsPath(x['path']) for x in installed if x['path'].endswith('vk_swiftshader.dll')), 'effective ICD path differs')
        need(len(r['commands']) == 2, 'missing Windows discovery commands')
        for index, command in enumerate(r['commands']):
            label, code = ('discovery', 0) if index == 0 else ('feature-rejection', 256)
            need(command['name'] == label and command['returncode'] == command['expected_native_returncode'] == code and
                 command['expected_logical_application_code'] == index and
                 PureWindowsPath(command['command'][0]).name == 'easy-diffusion.exe', 'Windows native launch/rejection code differs')
            out = self.read(group, subdir + '/' + label + '.stdout.log')
            err = self.read(group, subdir + '/' + label + '.stderr.log')
            need('WebGPU: Vulkan provider=native; library=vulkan-1.dll' in err, 'missing actual native loader trace')
            if index == 0:
                rows = [line.split('\t') for line in out.splitlines() if line.startswith('unavailable\t')]
                need(rows == [['unavailable', 'native', info['name'], reason]] and not re.search(r'^\d+\tWebGPU\t', out, re.M), 'discovery did not reject real SwiftShader')
            else:
                need('WEBGPU_DEVICE_TEST FAIL:' in err and 'lacks required ShaderF16; skipped' in err and
                     'WEBGPU_DEVICE_TEST PASS' not in out and 'WEBGPU_DEVICE_EXECUTION' not in out, 'crash/missing driver mistaken for feature rejection')
        return {'native_DLL_discovery': True, 'shader_f16_rejection': True, 'compute_verified': False,
                'DLL_hashes_scope': 'recorded on runner; installed DLL bytes are not retained here'}

    def config_unit(self, group, path):
        r = self.json(group, path)
        need(r['status'] == 'PASS' and r['checks'] == len(r['observations']) and r['checks'] == 38, 'configuration unit did not pass complete case list')
        expected_failure = 512 if group == 'windows' else 2
        observations = r['observations']
        # The three device-diagnostic invocations were added after the 35-case
        # draft. Check their actual argv/status, as well as the full ordered
        # sequence, rather than accepting an arbitrary list of 38 outcomes.
        # Temporary directory spelling differs between Windows/Linux runners.
        settings = observations[5]['arguments'][3]
        invalid = observations[31]['arguments'][3]
        path_type = PureWindowsPath if group == 'windows' else PurePosixPath
        settings_path, invalid_path = path_type(settings), path_type(invalid)
        need(settings_path.is_absolute() and settings_path.name == 'settings.json' and
             settings_path.parent.name == 'config' and settings_path.parent.parent.name.startswith('cosmo-config-') and
             invalid_path == settings_path.parent.parent / 'invalid.json',
             'configuration probe temporary paths differ')
        expected = [
            (['config', 'defaults'], 0),
            (['webgpu-device-test', '--provider', 'native'], 0),
            (['webgpu-device-test', '--device', 'WebGPU0', '--all'], 0),
            (['config', 'show'], expected_failure),
            (['config', 'init', '--config', 'config/settings.json'], 0),
            (['config', 'init', '--config', settings], expected_failure),
            (['webgpu-device-test', '--config', settings], 0),
            (['config', 'validate', '--config', settings], 0),
            (['sdkit', '--config', settings, '--backend', 'webgpu', '--provider', 'embedded',
              '--device', 'WebGPU7', '--port', '9001'], 0),
            (['config', 'show', '--config', settings], 0),
            (['image', '--config', 'first.json', '--provider', 'embedded', '--ckpt-dir', 'weights',
              '--prompt', '--device'], 0),
            (['image', '--config', 'first.json', '--prompt', '--device'], 0),
        ]
        for options in (['--backend', 'cuda'], ['--provider', 'other'], ['--device', '../bad'],
                        ['--port', '0'], ['--port', '65536'], ['--port', '1x'],
                        ['--backend', 'cpu', '--device', 'WebGPU0']):
            expected.append((['sdkit', '--config', 'invalid-first.json', *options], expected_failure))
        expected.append((['sdkit', '--config', settings, '--port', '9001', '--port', '9002'], expected_failure))
        expected.extend([(['sdkit', '--config', settings], 0)] * 3)
        expected.extend([(['sdkit', '--config', settings], expected_failure)] * 8)
        expected.extend([(['config', 'validate', '--config', invalid], expected_failure)] * 4)
        expected.extend([(['webgpu-test'], 0), (['--help'], 0), (['sdkit', '--help'], 0)])
        need([(x['arguments'], x['status']) for x in observations] == expected,
             'configuration observation sequence or native exit ABI differs')
        return {'checks': r['checks'], 'expected_negative_native_exit': expected_failure,
                'ordered_arguments_and_statuses_verified': True,
                'device_diagnostic_cases': 3,
                'device_diagnostic_contract': ['explicit native provider implies WebGPU without creating config',
                    'explicit WebGPU0 preserves --all without creating config',
                    'existing configuration preserves saved compute defaults'],
                'contract_evidence': 'PASS from hash-matched production verifier; report retains argv/status, not probe stdout or filesystem snapshots',
                'scope': 'configuration module probe; no HTTP or hardware'}

    def policy_unit(self, group, path):
        r = self.json(group, path)
        need(r['status'] == 'PASS' and r['artifact_sha256'] == digest(self.file('app', 'device-policy-test.exe')), 'policy artifact/status differs')
        need([x['scenario'] for x in r['cases']] == list(POLICY_CASES) and all(x['status'] == 'PASS' for x in r['cases']), 'missing policy cases')
        return {'cases': len(POLICY_CASES), 'scope': 'mock factory only; no native driver proof'}

    def lane_unit(self):
        group, subdir = 'diagnostics', 'native-main-executor-test'
        r = self.json(group, subdir + '/report.json')
        need(r['success'] is True and r['returncode'] == 0 and valid_sha(r['sha256_before']) and
             r['sha256_before'] == r['sha256_after'] and valid_sha(r['fixture_sha256']), 'native lane probe failed')
        out = self.read(group, subdir + '/stdout.log')
        err = self.read(group, subdir + '/stderr.log')
        for filename, key in [('stdout.log', 'stdout_sha256'), ('stderr.log', 'stderr_sha256')]:
            need(digest(self.file(group, subdir + '/' + filename)) == r[key], 'native lane raw hash differs')
        need(out.splitlines().count('NATIVE_MAIN_EXECUTOR checks=14 PASS') == 1 and
             re.search(r'^NATIVE_LANE_VULKAN extensions=[1-9][0-9]* tid=[1-9][0-9]*$', out, re.M), 'missing TLS/native Vulkan lane marker')
        need(err.count('NATIVE_INFERENCE_MAIN stack_bytes=') == 2 and err.count('NATIVE_HTTP_SERVICE stack_bytes=') == 2 and
             'NATIVE_INFERENCE_WORKER' not in err, 'main executor lane not exercised')
        loader = r.get('model_loader')
        if loader is not None:
            need(loader['checks'] == 9 and out.splitlines().count('NATIVE_MODEL_LOADER checks=9 PASS') == 1 and
                 re.search(r'^NATIVE_MODEL_LOADER_HOST extensions=[1-9][0-9]* tid=[1-9][0-9]*$', out, re.M),
                 'missing corrected native model-loader regression')
            need(all(loader[key] is True for key in ('native_host_tls_and_vulkan_enumeration',
                 'native_policy_runs_inline', 'embedded_policy_preserved', 'wrong_thread_rejected_before_work')),
                 'native loader policy/exception checks did not pass')
            legacy = r['legacy_loader_worker']
            need(legacy['fault_reproduced'] is True and legacy['returncode'] == legacy['expected_returncode'] == -11,
                 'legacy worker host-TLS fault not reproduced')
            for filename, key in [('legacy-worker.stdout.log', 'stdout_sha256'), ('legacy-worker.stderr.log', 'stderr_sha256')]:
                need(digest(self.file(group, subdir + '/' + filename)) == legacy[key], 'legacy raw log hash differs')
            old = self.read(group, subdir + '/legacy-worker.stderr.log')
            marker = re.search(r'^NATIVE_MODEL_LOADER_LEGACY worker_tid=(\d+) main_tid=(\d+) before_host_tls_call$', old, re.M)
            need(marker and marker.group(1) != marker.group(2) and 'Unexpected legacy worker host TLS return' not in old,
                 'legacy failure is not scoped to the off-main host TLS call')
        return {'checks': 14, 'model_loader_checks': 9 if loader is not None else None,
                'scope': 'native TLS/Vulkan enumeration; no shader proof', 'probe_bytes_retained': False}

    def adapter_unit(self):
        log = self.read('diagnostics', 'adapter-enumeration-test.log')
        match = re.search(r'test result: ok\. (\d+) passed; 0 failed;', log)
        need(match and int(match.group(1)) > 0 and 'FAILED' not in log, 'adapter enumeration Rust test did not pass')
        return {'tests': int(match.group(1)), 'scope': 'adapter reference/list handling units; no physical GPU'}

    def run(self):
        self.record('application-package', self.package)
        if self.groups[-1]['status'] == 'failed':
            return
        self.record('linux-isolated-runtime', self.runtime_gate, 'linux', 'results-linux-isolated', True)
        self.record('windows-native-runtime', self.runtime_gate, 'windows', 'results-windows', False)
        self.record('linux-shell-bootstrap', self.bootstrap_gate)
        for group in ('linux', 'windows'):
            for backend in ('cpu', 'webgpu'):
                self.record(group + '-image-' + backend, self.image_gate, group, backend)
            self.record(group + '-inference-api', self.api_gate, group)
            self.record(group + '-configuration-http-restart', self.configuration_gate, group)
        self.record('linux-native-device', self.device_gate, False)
        self.record('linux-native-embedded-coexistence', self.device_gate, True)
        self.record('linux-native-inference-api', self.api_gate, 'linux-native', True)
        self.record('windows-real-DLL-discovery-and-rejection', self.windows_discovery)
        self.record('linux-configuration-unit', self.config_unit, 'diagnostics', 'config-test/report.json')
        self.record('windows-configuration-unit', self.config_unit, 'windows', 'config-report.json')
        self.record('linux-device-policy-unit', self.policy_unit, 'diagnostics', 'device-policy-report.json')
        self.record('windows-device-policy-unit', self.policy_unit, 'windows', 'device-policy-report.json')
        self.record('linux-native-TLS-main-lane-unit', self.lane_unit)
        self.record('rust-adapter-enumeration-unit', self.adapter_unit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('app', 'linux', 'windows', 'linux-native', 'diagnostics'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-source', required=True, help='Exact 40-character clean source commit')
    parser.add_argument('--expected-sha256', help='Optional external build-job application digest')
    parser.add_argument('--parser-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'tests')
    parser.add_argument('--output', type=Path, help='Write inspection JSON here; otherwise stdout')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.expected_source):
        parser.error('--expected-source must be a full lowercase commit SHA')
    if args.expected_sha256 and not valid_sha(args.expected_sha256):
        parser.error('--expected-sha256 must be a full lowercase SHA-256')
    review = Inspector(args)
    review.run()
    passed = len(review.groups) == 22 and all(x['status'] == 'passed' for x in review.groups)
    result = {'status': 'passed' if passed else 'failed', 'expected_source': args.expected_source,
              'application_sha256': getattr(review, 'app_sha', None), 'groups': review.groups,
              'files': [review.inventory[k] for k in sorted(review.inventory)],
              'scope': {'offline_only': True, 'app_or_model_executed_by_inspector': False,
                        'network_used': False, 'physical_GPU_verified': False,
                        'native_Linux_driver': 'host Mesa CPU/software; actual graphs, llama and HTTP image inference',
                        'native_Windows_driver': 'installed Chrome SwiftShader DLL discovery and f16 rejection only',
                        'external_diffusion_checkpoint_bytes_retained': False, 'DLL_bytes_retained': False,
                        'historical_logs_without_recorded_hashes': 'reparsed and freshly inventoried, not retroactively hash-bound to runner',
                        'job_completion_and_artifact_download_authenticity': 'must be verified separately from trusted CI metadata',
                        'not_claimed': ['physical GPU inference', 'image quality', 'full operation/model parity', 'training parity', 'browser execution']}}
    payload = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(json.dumps({'status': result['status'], 'groups': len(review.groups), 'report': str(args.output),
                          'failures': [x for x in review.groups if x['status'] == 'failed']}))
    else:
        print(payload, end='')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
