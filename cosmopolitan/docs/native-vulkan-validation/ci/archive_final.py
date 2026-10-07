#!/usr/bin/env python3
"""Archive completed native-provider CI evidence, offline and without executing it.

Run only after the final inspection and all four CI jobs pass. The source is an
extracted artifact collection containing application/, diagnostics/, linux/,
windows/, linux-native/, final REST snapshots, download-verification records,
and inspection.json. The destination must not already exist.

The archived inspect_results.py needs an explicit --parser-dir pointing to
/path/to/matching/checkout/cosmopolitan/tests; its out/ layout default does not
apply inside this archive. This helper never imports or runs that inspector.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import zipfile

PORT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE = '1730424270e35ff4a06bd978654c9990cfd5777e'
EXPECTED_RUN = 37568302821
EXPECTED_APP = '7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9'
EXPECTED_APP_BYTES = 190491826
JOB_NAMES = {
    'Build one application with shared GGML and both Vulkan providers',
    'Linux application in an empty filesystem',
    'Windows native PE with the identical application',
    'Linux native Vulkan loader and software-driver inference',
}
ARTIFACT_NAMES = {
    'application': 'easy-diffusion-cosmopolitan-x86_64',
    'diagnostics': 'cosmopolitan-build-diagnostics',
    'linux': 'cosmopolitan-linux-results',
    'windows': 'cosmopolitan-windows-results',
    'linux-native': 'cosmopolitan-linux-native-results',
}
INSPECTION_GROUPS = {
    'application-package', 'linux-isolated-runtime', 'windows-native-runtime',
    'linux-shell-bootstrap', 'linux-image-cpu', 'linux-image-webgpu',
    'linux-inference-api', 'linux-configuration-http-restart',
    'windows-image-cpu', 'windows-image-webgpu', 'windows-inference-api',
    'windows-configuration-http-restart', 'linux-native-device',
    'linux-native-embedded-coexistence', 'linux-native-inference-api',
    'windows-real-DLL-discovery-and-rejection', 'linux-configuration-unit',
    'windows-configuration-unit', 'linux-device-policy-unit',
    'windows-device-policy-unit', 'linux-native-TLS-main-lane-unit',
    'rust-adapter-enumeration-unit',
}
APP_FILES = (
    'BUILD.json', 'SYMBOLS.json', 'PIN.json', 'CONFIG_PROBE_BUILD.json',
    'DEVICE_POLICY_BUILD.json', 'software-webgpu/LINK.json',
    'verify_runtime.py', 'verify_bootstrap.py', 'verify_inference.py',
    'verify_inference_api.py', 'verify_webgpu_device.py',
    'verify_windows_native_discovery.py', 'verify_configuration_api.py',
    'verify_config.py', 'verify_device_policy.py', 'fetch_inference_model.py',
)
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def safe_file(root, relative):
    rel = PurePosixPath(relative)
    require(relative and not rel.is_absolute() and not PureWindowsPath(relative).is_absolute()
            and '..' not in rel.parts and '\\' not in relative, 'unsafe relative path: ' + relative)
    path = root.joinpath(*rel.parts)
    current = root
    for component in rel.parts:
        current /= component
        require(not current.is_symlink(), 'symlink evidence is not accepted: ' + str(current))
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), 'missing/escaped evidence: ' + str(path))
    return path


def document(root, relative):
    return json.loads(safe_file(root, relative).read_text())


def validate(root):
    run = document(root, 'run.json')
    require(run['id'] == EXPECTED_RUN and run['head_sha'] == EXPECTED_SOURCE and
            run['status'] == 'completed' and run['conclusion'] == 'success', 'run is not the completed expected successful run')
    jobs = document(root, 'jobs.json')
    require(jobs['total_count'] == len(jobs['jobs']) == 4 and
            {job['name'] for job in jobs['jobs']} == JOB_NAMES, 'missing, duplicate, or unexpected CI jobs')
    for job in jobs['jobs']:
        require(job['run_id'] == EXPECTED_RUN and job['head_sha'] == EXPECTED_SOURCE and
                job['status'] == 'completed' and job['conclusion'] == 'success' and
                job.get('run_attempt') == run['run_attempt'], 'job does not belong to this successful run attempt')
        require(job.get('started_at') and job.get('completed_at') and job.get('html_url'), 'job provenance/timestamps missing')
        require(all(step['status'] == 'completed' and step.get('conclusion') in ('success', 'skipped')
                    for step in job['steps']), 'unfinished or failed job step')
    manifest = document(root, 'application/BUILD.json')
    require(manifest['source']['commit'] == EXPECTED_SOURCE and manifest['source']['dirty'] is False,
            'packaged source is wrong or dirty')
    require(manifest['executable']['sha256'] == EXPECTED_APP and manifest['executable']['bytes'] == EXPECTED_APP_BYTES,
            'packaged executable identity differs')
    inspection = document(root, 'inspection.json')
    require(inspection['status'] == 'passed' and inspection['expected_source'] == EXPECTED_SOURCE and
            inspection['application_sha256'] == EXPECTED_APP, 'inspection did not pass the exact final artifact')
    require(len(inspection['groups']) == len(INSPECTION_GROUPS) and
            {group['group'] for group in inspection['groups']} == INSPECTION_GROUPS and
            all(group['status'] == 'passed' and not group.get('error') for group in inspection['groups']),
            'not all required inspection groups passed')
    require(inspection['scope']['physical_GPU_verified'] is False and inspection['scope']['offline_only'] is True,
            'inspection overstates its scope')
    inventory = inspection['files']
    require(inventory and len({row['path'] for row in inventory}) == len(inventory), 'inspection inventory is empty or duplicated')
    for row in inventory:
        prefix, slash, tail = row['path'].partition('/')
        require(slash and prefix in ('app', 'linux', 'windows', 'linux-native', 'diagnostics'), 'unknown inspection input group')
        group = 'application' if prefix == 'app' else prefix
        path = safe_file(root, group + '/' + tail)
        require(info(path) == {'bytes': row['bytes'], 'sha256': row['sha256']}, 'evidence changed since inspection: ' + row['path'])
    require(any(row['path'] == 'app/easy-diffusion.exe' and row['sha256'] == EXPECTED_APP and
                row['bytes'] == EXPECTED_APP_BYTES for row in inventory), 'inspection lacks actual application byte hash')
    app_review = document(root, 'application-review.json')
    require(app_review['status'] == 'passed' and app_review['details']['source']['commit'] == EXPECTED_SOURCE and
            app_review['details']['app_sha256'] == EXPECTED_APP, 'package-only review identity differs')
    artifacts = document(root, 'artifacts.json')
    require(artifacts['total_count'] == len(artifacts['artifacts']), 'artifact snapshot is incomplete')
    records = {}
    for group, name in ARTIFACT_NAMES.items():
        matches = [item for item in artifacts['artifacts'] if item['name'] == name]
        require(len(matches) == 1, 'missing or duplicate artifact: ' + name)
        item = matches[0]
        require(item['workflow_run']['id'] == EXPECTED_RUN and item['workflow_run']['head_sha'] == EXPECTED_SOURCE,
                'artifact belongs to a different source/run: ' + name)
        receipt = document(root, group + '-download-verification.json')
        require(Path(receipt['extracted']).resolve() == (root / group).resolve(), 'download receipt extraction path differs')
        archive_path = Path(receipt['path']).resolve()
        require(archive_path.is_relative_to((root / 'downloads').resolve()) and archive_path.is_file() and
                not Path(receipt['path']).is_symlink(), 'download ZIP is not the retained local download')
        require(info(archive_path) == {'bytes': receipt['bytes'], 'sha256': receipt['sha256']}, 'download ZIP bytes changed')
        require(item['digest'] == 'sha256:' + receipt['sha256'] and item['size_in_bytes'] == receipt['bytes'],
                'download ZIP does not match GitHub artifact digest/size')
        records[group] = {'artifact_id': item['id'], 'name': name, 'archive': archive_path,
                          'sha256': receipt['sha256'], 'bytes': receipt['bytes']}
    safe_file(root, 'build-job.log')
    return run, jobs, records


def selection(root):
    files = []
    for name in APP_FILES:
        files.append(('application/' + name, safe_file(root, 'application/' + name), 'application', name))
    for group in ('diagnostics', 'linux', 'windows', 'linux-native'):
        directory = root / group
        require(directory.is_dir() and not directory.is_symlink(), 'missing evidence directory: ' + group)
        allowed = {'.json', '.log', '.txt', '.yaml'} if group == 'diagnostics' else {'.json', '.log', '.txt', '.png'}
        count = 0
        for path in sorted(directory.rglob('*')):
            require(not path.is_symlink(), 'symlink in evidence tree: ' + str(path))
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            relative = path.relative_to(directory).as_posix()
            safe_file(root, group + '/' + relative)
            files.append((group + '/' + relative, path, group, relative))
            count += 1
        require(count > 0, 'empty selected evidence directory: ' + group)
    for name in ('run.json', 'jobs.json', 'artifacts.json', 'inspection.json', 'application-review.json',
                 'build-job.log', *(group + '-download-verification.json' for group in ARTIFACT_NAMES)):
        files.append((name, safe_file(root, name), None, None))
    # Additional full job logs are optional; runtime stdout/stderr are mandatory through inspection.
    for name in ('linux-job.log', 'windows-job.log', 'linux-native-job.log'):
        if (root / name).exists():
            files.append((name, safe_file(root, name), None, None))
    for name in ('inspect_results.py', 'archive_final.py'):
        script = Path(__file__).resolve().with_name(name)
        require(script.is_file() and not script.is_symlink(), 'missing local inspection/archive recipe: ' + name)
        files.append((name, script, None, None))
    require(len({entry[0] for entry in files}) == len(files), 'duplicate archive destination')
    require(all(path.stat().st_size <= MAX_FILE_BYTES for _, path, _, _ in files), 'selected evidence exceeds per-file small-evidence limit')
    require(sum(path.stat().st_size for _, path, _, _ in files) <= MAX_ARCHIVE_BYTES, 'selected evidence exceeds total small-evidence limit')
    return files


def archive(root, destination):
    require(not destination.exists(), 'destination already exists; refusing to replace prior evidence')
    run, jobs, receipts = validate(root)
    files = selection(root)
    # All publication gates have passed before a staging tree or destination is created.
    with tempfile.TemporaryDirectory(prefix='native-evidence-stage-', dir=root) as temporary:
        stage = Path(temporary) / 'ci'
        stage.mkdir()
        zips = {group: zipfile.ZipFile(item['archive']) for group, item in receipts.items()}
        try:
            for relative, source, group, member in files:
                recorded = info(source)
                if group:
                    entry = zips[group].getinfo(member)
                    require(not entry.is_dir() and entry.file_size == recorded['bytes'], 'ZIP member size differs: ' + relative)
                    h = hashlib.sha256()
                    with zips[group].open(member) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block)
                    require(h.hexdigest() == recorded['sha256'], 'extracted evidence differs from original ZIP: ' + relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                require(info(target) == recorded, 'copy integrity failure: ' + relative)
        finally:
            for handle in zips.values():
                handle.close()
        (stage / '.gitattributes').write_bytes(b'* -text\n')
        provenance = {
            'schema': 1, 'run_id': EXPECTED_RUN, 'run_url': run['html_url'], 'run_attempt': run['run_attempt'],
            'source_commit': EXPECTED_SOURCE, 'application_sha256': EXPECTED_APP,
            'application_bytes': EXPECTED_APP_BYTES, 'publication_gate': 'all four jobs and all 22 inspection groups passed',
            'jobs': [{'id': j['id'], 'name': j['name'], 'url': j['html_url']} for j in jobs['jobs']],
            'artifacts': {group: {k: v for k, v in receipt.items() if k != 'archive'} for group, receipt in receipts.items()},
            'preservation': 'Original file bytes; * -text disables Git newline conversion. FILES.json excludes itself.',
            'excluded': ['executables', 'ELF loaders', 'debug binaries', 'model checkpoints', 'static/shared libraries', 'download ZIPs'],
            'scope': {'physical_GPU_verified': False, 'native_Linux': 'host software Vulkan driver, actual graph/model/API execution',
                      'native_Windows': 'installed Chrome SwiftShader DLL discovery and ShaderF16 rejection; no native compute proof',
                      'software_fallback': 'embedded lavapipe; Linux isolation remains a separate gate',
                      'image_quality_or_full_training_parity': False},
            'inspection_reproduction': 'Use archived inspect_results.py with the original five extracted artifacts, '
                '--expected-source ' + EXPECTED_SOURCE + ' --expected-sha256 ' + EXPECTED_APP +
                ' --parser-dir /path/to/matching/checkout/cosmopolitan/tests. Executables and download ZIPs are intentionally not archived.',
        }
        (stage / 'ARCHIVE.json').write_text(json.dumps(provenance, indent=2) + '\n')
        inventory = [{'path': path.relative_to(stage).as_posix(), **info(path)}
                     for path in sorted(stage.rglob('*')) if path.is_file()]
        (stage / 'FILES.json').write_text(json.dumps({'schema': 1, 'run_id': EXPECTED_RUN,
            'source_commit': EXPECTED_SOURCE, 'application_sha256': EXPECTED_APP, 'hash_algorithm': 'SHA-256',
            'files': inventory}, indent=2) + '\n')
        for item in inventory:
            require(info(stage / item['path']) == {'bytes': item['bytes'], 'sha256': item['sha256']}, 'staged inventory mismatch')
        require(not destination.exists(), 'destination appeared during staging; refusing to replace it')
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(destination)
    return {'status': 'archived', 'destination': str(destination), 'run_id': EXPECTED_RUN,
            'source_commit': EXPECTED_SOURCE, 'application_sha256': EXPECTED_APP,
            'files_in_inventory': len(inventory), 'bytes_in_inventory': sum(item['bytes'] for item in inventory),
            'physical_GPU_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PORT / 'out/native-final-ci')
    args = parser.parse_args()
    root = args.input_dir.resolve()
    destination = PORT / 'docs/native-vulkan-validation/ci'
    try:
        result = archive(root, destination)
    except Exception as error:
        print(json.dumps({'status': 'not-archived', 'error': str(error), 'destination': str(destination)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
