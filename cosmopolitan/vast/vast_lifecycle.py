#!/usr/bin/env python3
"""Bound the existing workflow's Vast CLI lifecycle; never retry creation.

Vast CLI 1.8.3 drops destroy --raw's returned body. Destruction therefore needs
a complete `show instances --all --raw` absence observation AND an explicit
`show instance ID --raw` {"instances": null} response. Both are authenticated.
Only this module's CLI entry point can contact Vast. Tests inject an offline CLI.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

VERSION = '1.8.3'


class LifecycleError(RuntimeError):
    pass


def identifier(value):
    if isinstance(value, bool) or not re.fullmatch(r'[1-9][0-9]*', str(value)):
        raise LifecycleError('Invalid instance/offer identifier')
    return int(value)


def rows_from(result):
    if result['returncode'] != 0 or result.get('timed_out'):
        raise LifecycleError('Instance listing did not complete successfully')
    try:
        rows = json.loads(result['stdout'])
    except (ValueError, TypeError) as error:
        raise LifecycleError('Instance listing was not JSON') from error
    if not isinstance(rows, list):
        raise LifecycleError('Expected the pinned CLI complete instance list')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or 'label' not in row:
            raise LifecycleError('Incomplete instance record')
        current = identifier(row.get('id'))
        if current in seen or not (row['label'] is None or isinstance(row['label'], str)):
            raise LifecycleError('Invalid or duplicate instance record')
        seen.add(current)
    return rows


class Cli:
    def __init__(self, executable='vastai'):
        self.executable = executable
        self.environment = dict(os.environ)
        key = self.environment.get('VASTAI_KEY')
        if not key:
            raise LifecycleError('VASTAI_KEY must be supplied by the approved workflow')
        self.environment['VAST_API_KEY'] = key
        # No key enters argv, captured diagnostics, or retained evidence.
        self.environment.pop('VASTAI_KEY', None)

    def call(self, arguments, timeout):
        with subprocess.Popen([self.executable, *arguments], env=self.environment,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
            timed_out = False
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                stdout, _ = process.communicate()
        return {'returncode': process.returncode, 'timed_out': timed_out, 'stdout': stdout}


class Lifecycle:
    def __init__(self, cli, evidence, *, github_output=None, clock=time.monotonic,
                 sleep=time.sleep, poll_seconds=3):
        self.cli, self.evidence = cli, Path(evidence)
        self.github_output, self.clock, self.sleep = github_output, clock, sleep
        self.poll_seconds = poll_seconds
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.events = []

    def save(self, name, data):
        destination = self.evidence / name
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        os.replace(temporary, destination)

    def output(self, name, value):
        if self.github_output:
            with Path(self.github_output).open('a') as output:
                output.write(f'{name}={value}\n')

    def version(self):
        result = self.cli.call(['--version'], 10)
        if result['returncode'] != 0 or result.get('timed_out') or not re.fullmatch(
                r'(?:vastai\s+)?1\.8\.3\s*', result['stdout'].strip()):
            raise LifecycleError('Expected exactly Vast CLI 1.8.3')

    def listing(self, timeout=20):
        # No --limit/next-token: --all explicitly traverses every page.
        result = self.cli.call(['show', 'instances', '--all', '--raw', '--retry', '1'], timeout)
        return rows_from(result)

    def explicit_absence(self, instance_id, timeout=20):
        result = self.cli.call(['show', 'instance', str(instance_id), '--raw', '--retry', '1'], timeout)
        if result['returncode'] != 0 or result.get('timed_out'):
            raise LifecycleError('Exact-instance absence query did not complete successfully')
        try:
            payload = json.loads(result['stdout'])
        except (ValueError, TypeError) as error:
            raise LifecycleError('Exact-instance absence query was not JSON') from error
        # CLI 1.8.3 emits precisely this object only when the API's explicit
        # instances member is null. Missing members raise inside the CLI. This
        # complements its list path, which can normalize a malformed API object
        # without an instances member into an empty list.
        return payload == {'instances': None}

    def publish_id(self, record, instance_id):
        instance_id = identifier(instance_id)
        record['instance_id'] = instance_id
        self.save('creation.json', record)
        (self.evidence / 'instance-id.txt').write_text(str(instance_id) + '\n')
        self.output('instance_id', instance_id)
        return instance_id

    def create(self, *, offer_id, image, label, capabilities, disk_gb=20,
               create_timeout=35, reconcile_timeout=45):
        offer_id = identifier(offer_id)
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', label):
            raise LifecycleError('Expected a unique simple run/attempt label')
        if not image or any(c.isspace() for c in image) or image.startswith('-'):
            raise LifecycleError('Invalid image reference')
        if not capabilities or not set(capabilities.split(',')) <= {
                'compute', 'utility', 'graphics', 'display', 'video', 'compat32'}:
            raise LifecycleError('Invalid NVIDIA driver capabilities')
        if not 1 <= disk_gb <= 1000:
            raise LifecycleError('Invalid disk allocation')
        self.version()
        rows = self.listing()
        if any(row['label'] == label for row in rows):
            raise LifecycleError('Run label already exists; creation refused')
        record = {'schema': 1, 'cli_version': VERSION, 'label': label,
                  'offer_id': offer_id, 'image': image, 'disk_gb': disk_gb,
                  'driver_capabilities': capabilities, 'label_absent_before_create': True,
                  'creation_attempted': True, 'status': 'creation-started'}
        self.save('creation.json', record)
        self.output('creation_attempted', 'true')
        result = self.cli.call(['create', 'instance', str(offer_id), '--image', image,
                               '--disk', str(disk_gb), '--ssh', '--direct', '--cancel-unavail',
                               '--env', '-e NVIDIA_DRIVER_CAPABILITIES=' + capabilities,
                               '--label', label, '--raw', '--retry', '1'], create_timeout)
        record['create_returncode'] = result['returncode']
        record['create_timed_out'] = result.get('timed_out', False)
        try:
            response = json.loads(result['stdout'])
        except (ValueError, TypeError):
            response = None
        if isinstance(response, dict):
            try:
                instance_id = identifier(response.get('new_contract'))
            except LifecycleError:
                instance_id = None
            if instance_id is not None:
                # Save before success validation so always-cleanup owns the ID
                # even when the response is inconsistent or the CLI timed out.
                self.publish_id(record, instance_id)
                if (response.get('success') is True and result['returncode'] == 0
                        and not result.get('timed_out')):
                    record['status'] = 'created'
                    self.save('creation.json', record)
                    return record
                record['status'] = 'creation-response-failed-id-preserved'
                self.save('creation.json', record)
                raise LifecycleError('Creation was not successful; recovered ID must be cleaned up')
        # A failed/late/malformed response is NOT evidence that no rental exists.
        # Never issue another create. Reconcile only the unique, prechecked label.
        deadline = self.clock() + reconcile_timeout
        record['reconciliation'] = []
        while self.clock() < deadline:
            try:
                rows = self.listing(min(20, max(0.1, deadline - self.clock())))
                matches = [identifier(row['id']) for row in rows if row['label'] == label]
                record['reconciliation'].append({'matching_ids': matches})
                if len(matches) == 1:
                    record['status'] = 'created-response-reconciled'
                    self.publish_id(record, matches[0])
                    return record
                if len(matches) > 1:
                    break
            except LifecycleError as error:
                record['reconciliation'].append({'error': str(error)})
            self.save('creation.json', record)
            self.sleep(min(self.poll_seconds, max(0, deadline - self.clock())))
        record['status'] = 'creation-UNCONFIRMED'
        self.save('creation.json', record)
        raise LifecycleError('Creation is UNCONFIRMED; do not create again. Inspect the exact run label.')

    def destroy(self, *, instance_id=None, label=None, timeout=150):
        self.version()
        deadline = self.clock() + timeout
        if instance_id is None:
            # The workflow may have been interrupted before GITHUB_OUTPUT got
            # its ID. Recover only a label whose pre-creation absence we recorded.
            record = json.loads((self.evidence / 'creation.json').read_text())
            if (not label or record.get('label') != label or
                    record.get('label_absent_before_create') is not True or
                    record.get('creation_attempted') is not True):
                raise LifecycleError('No safe creation record for label reconciliation')
            if record.get('instance_id') is not None:
                instance_id = identifier(record['instance_id'])
            else:
                while self.clock() < deadline:
                    try:
                        rows = self.listing(min(20, max(0.1, deadline - self.clock())))
                        matches = [identifier(row['id']) for row in rows if row['label'] == label]
                        if len(matches) > 1:
                            raise LifecycleError('Multiple IDs share this run label; cleanup is UNCONFIRMED')
                        if matches:
                            instance_id = self.publish_id(record, matches[0])
                            break
                    except LifecycleError as error:
                        self.events.append({'operation': 'reconcile', 'error': str(error)})
                    self.sleep(min(self.poll_seconds, max(0, deadline - self.clock())))
                if instance_id is None:
                    self.save('cleanup.json', {'status': 'UNCONFIRMED', 'label': label,
                                              'reason': 'No ID recovered after ambiguous creation',
                                              'events': self.events})
                    raise LifecycleError('Unknown creation outcome; absence of a label is not proof of no delayed rental')
        instance_id = identifier(instance_id)
        record = {'schema': 1, 'cli_version': VERSION, 'instance_id': instance_id,
                  'status': 'UNCONFIRMED', 'events': self.events}
        self.save('cleanup.json', record)
        attempts = 0
        next_destroy = self.clock()
        while self.clock() < deadline:
            # The CLI return code/body is diagnostic only. Its destroy command
            # currently discards the API's success/failure JSON response.
            if attempts < 3 and self.clock() >= next_destroy:
                result = self.cli.call(['destroy', 'instance', str(instance_id), '-y',
                                        '--raw', '--retry', '1'],
                                       min(30, max(0.1, deadline - self.clock())))
                self.events.append({'operation': 'destroy', 'returncode': result['returncode'],
                                    'timed_out': result.get('timed_out', False)})
                attempts += 1
                next_destroy = self.clock() + 15
            try:
                rows = self.listing(min(20, max(0.1, deadline - self.clock())))
                present = any(identifier(row['id']) == instance_id for row in rows)
                self.events.append({'operation': 'complete-instance-list', 'target_present': present})
                if not present:
                    absent = self.explicit_absence(instance_id, min(20, max(0.1, deadline - self.clock())))
                    self.events.append({'operation': 'exact-instance-query', 'explicit_absence': absent})
                    if absent:
                        record['status'] = 'destroyed-absence-confirmed'
                        record['proof'] = ('Exact ID absent from complete authenticated instance list; '
                                           'separate exact-ID query explicitly returned instances:null')
                        self.save('cleanup.json', record)
                        return record
            except LifecycleError as error:
                self.events.append({'operation': 'complete-instance-list', 'error': str(error)})
            self.save('cleanup.json', record)
            self.sleep(min(self.poll_seconds, max(0, deadline - self.clock())))
        self.save('cleanup.json', record)
        raise LifecycleError(f'Destruction of instance {instance_id} is UNCONFIRMED; billing may continue')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', default='vastai')
    parser.add_argument('--evidence-dir', required=True, type=Path)
    sub = parser.add_subparsers(dest='operation', required=True)
    create = sub.add_parser('create')
    create.add_argument('--offer-id', required=True)
    create.add_argument('--image', required=True)
    create.add_argument('--label', required=True)
    create.add_argument('--driver-capabilities', required=True)
    create.add_argument('--disk-gb', type=int, default=20)
    destroy = sub.add_parser('destroy')
    destroy.add_argument('--instance-id')
    destroy.add_argument('--label')
    args = parser.parse_args()
    try:
        lifecycle = Lifecycle(Cli(args.cli), args.evidence_dir,
                              github_output=os.environ.get('GITHUB_OUTPUT'))
        if args.operation == 'create':
            result = lifecycle.create(offer_id=args.offer_id, image=args.image, label=args.label,
                                      capabilities=args.driver_capabilities, disk_gb=args.disk_gb)
        else:
            result = lifecycle.destroy(instance_id=args.instance_id, label=args.label)
        print(json.dumps(result, indent=2))
        return 0
    except (LifecycleError, OSError, ValueError) as error:
        print('Vast lifecycle failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
