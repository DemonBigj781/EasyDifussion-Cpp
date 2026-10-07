#!/usr/bin/env python3
"""Model-free configuration contract checks; accepts the focused config_probe.

The probe links production config.cpp. This does not claim HTTP, driver, or
inference coverage. Pass an APE loader on Linux if the host has no APE binfmt.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--loader', type=Path)
    args = parser.parse_args()
    command = ([str(args.loader.resolve())] if args.loader else []) + [str(args.probe.resolve())]
    observations = []
    with tempfile.TemporaryDirectory(prefix='cosmo-config-') as directory:
        root = Path(directory)
        def portable(path):
            text = str(path)
            if os.name == 'nt':
                text = text.replace('\\', '/')
                if len(text) >= 3 and text[1:3] == ':/':
                    text = '/' + text[0] + text[2:]
            return text

        env = dict(os.environ)
        env.pop('COSMO_CONFIG_TEST_UPDATE', None)
        env.pop('COSMO_CONFIG_TEST_OPTIONS', None)

        def run(*values, code=0, update=None, options=None):
            process_env = dict(env)
            if update is not None:
                process_env['COSMO_CONFIG_TEST_UPDATE'] = json.dumps(update)
            if options is not None:
                process_env['COSMO_CONFIG_TEST_OPTIONS'] = json.dumps(options)
            result = subprocess.run(command + list(values), cwd=root, env=process_env,
                                    text=True, capture_output=True, timeout=20)
            # The pinned Cosmopolitan Windows exit ABI exposes logical status << 8.
            expected = code << 8 if os.name == 'nt' else code
            assert result.returncode == expected, (values, result.returncode, expected, result.stdout, result.stderr)
            observations.append({'arguments': list(values), 'status': result.returncode})
            return result

        def document(result):
            # config show/init are pretty-printed; normal probe documents precede ARG records.
            text = result.stdout.split('\nARG ', 1)[0]
            return json.loads(text)

        defaults = json.loads(run('config', 'defaults').stdout)
        assert defaults['compute'] == {'backend': 'cpu', 'provider': 'auto', 'device': 'auto'}
        assert not (root / 'easy-diffusion.json').exists()
        diagnostic = document(run('webgpu-device-test', '--provider', 'native'))
        assert diagnostic['effective']['compute']['provider'] == 'native'
        assert diagnostic['effective']['compute']['backend'] == 'webgpu'
        assert 'ARG --all' in run('webgpu-device-test', '--device', 'WebGPU0', '--all').stdout
        assert not (root / 'easy-diffusion.json').exists()
        run('config', 'show', code=2)
        initialized = document(run('config', 'init', '--config', 'config/settings.json'))
        config = root / 'config/settings.json'
        original = config.read_bytes()
        assert initialized['effective']['models']['checkpoint_dir'] == portable(root / 'config/models/checkpoints')
        run('config', 'init', '--config', str(config), code=2)
        assert config.read_bytes() == original
        assert document(run('webgpu-device-test', '--config', str(config)))['saved']['compute'] == defaults['compute']
        run('config', 'validate', '--config', str(config))
        changed = document(run('sdkit', '--config', str(config), '--backend', 'webgpu', '--provider', 'embedded', '--device', 'WebGPU7', '--port', '9001'))
        assert changed['effective']['compute'] == {'backend': 'webgpu', 'provider': 'embedded', 'device': 'WebGPU7'}
        assert changed['saved']['compute'] == defaults['compute'] and changed['restart_required']
        assert config.read_bytes() == original
        assert document(run('config', 'show', '--config', str(config)))['effective']['compute'] == defaults['compute']
        first = document(run('image', '--config', 'first.json', '--provider', 'embedded', '--ckpt-dir', 'weights', '--prompt', '--device'))
        assert first['effective']['models']['checkpoint_dir'] == portable(root / 'weights')
        assert json.loads((root / 'first.json').read_text())['compute']['provider'] == 'auto'
        assert 'ARG --prompt\nARG --device' in run('image', '--config', 'first.json', '--prompt', '--device').stdout
        for args in [('--backend', 'cuda'), ('--provider', 'other'), ('--device', '../bad'), ('--port', '0'), ('--port', '65536'), ('--port', '1x'), ('--backend', 'cpu', '--device', 'WebGPU0')]:
            run('sdkit', '--config', 'invalid-first.json', *args, code=2)
            assert not (root / 'invalid-first.json').exists()
        run('sdkit', '--config', str(config), '--port', '9001', '--port', '9002', code=2)
        updated = document(run('sdkit', '--config', str(config), update={'compute': {'backend': 'webgpu', 'provider': 'embedded', 'device': 'auto'}}))
        assert updated['saved']['compute']['backend'] == 'webgpu'
        assert updated['effective']['compute']['backend'] == 'cpu' and updated['restart_required']
        restarted = document(run('sdkit', '--config', str(config)))
        assert restarted['effective']['compute']['backend'] == 'webgpu' and not restarted['restart_required']
        live = document(run('sdkit', '--config', str(config), options={'sd_model_checkpoint': 'named.safetensors'}))
        assert live['effective']['options']['sd_model_checkpoint'] == live['saved']['options']['sd_model_checkpoint'] == 'named.safetensors'
        before = config.read_bytes()
        for update in [{'unknown': 1}, {'compute': {'backend': 'cuda'}}, {'compute': {'provider': 'invented'}}, {'options': {'sd_model_checkpoint': 'ignored'}}]:
            run('sdkit', '--config', str(config), update=update, code=2)
            assert config.read_bytes() == before
        for options in [{'unknown': 1}, {'live_previews_enable': True}, {'forge_additional_modules': ['a']}, {'sd_model_checkpoint': 4}]:
            run('sdkit', '--config', str(config), options=options, code=2)
            assert config.read_bytes() == before
        invalid = root / 'invalid.json'
        for text in ['{bad', '{"schema":1,"schema":1}', json.dumps({'schema': 1}), before.decode().replace('"port": 8188', '"port": "8188"')]:
            invalid.write_text(text)
            run('config', 'validate', '--config', str(invalid), code=2)
            assert invalid.read_text() == text
        # Invalid stored config cannot interfere with diagnostics/help.
        (root / 'easy-diffusion.json').write_text('invalid saved JSON')
        diagnostic = run('webgpu-test')
        assert 'POLICY cpu embedded auto' in diagnostic.stdout
        run('--help')
        run('sdkit', '--help')
        assert (root / 'easy-diffusion.json').read_text() == 'invalid saved JSON'
        assert not list(root.rglob('*.tmp.*')), 'temporary files leaked'
    print(json.dumps({'status': 'PASS', 'scope': 'production configuration module; no HTTP/model/driver execution', 'checks': len(observations), 'observations': observations}, indent=2))


if __name__ == '__main__':
    main()
