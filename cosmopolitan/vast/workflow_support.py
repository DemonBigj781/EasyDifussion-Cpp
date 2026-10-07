#!/usr/bin/env python3
"""Offline validation used by the existing Vast workflow. Never calls Vast."""

import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def money(value, name, *, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{name} is not a decimal amount')
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f'{name} is not a decimal amount') from error
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError(f'{name} must be finite and ' + ('positive' if positive else 'nonnegative'))
    return amount


def approval_gate(approval, offer_id, base_image, max_hourly_usd, backend, capabilities, timeout):
    if approval != 'APPROVE':
        raise ValueError('Literal APPROVE is required. No Vast API call has been made.')
    if not re.fullmatch(r'[0-9]+', offer_id):
        raise ValueError('An explicitly selected numeric offer ID is required')
    if not re.fullmatch(r'[a-z0-9][a-z0-9./:@_-]*', base_image):
        raise ValueError('A single valid base-image reference is required')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', backend):
        raise ValueError('Backend must be a simple image-tag component')
    allowed = {'compute', 'utility', 'graphics', 'display', 'video', 'compat32'}
    if not capabilities or not set(capabilities.split(',')) <= allowed:
        raise ValueError('Unknown NVIDIA driver capability')
    if not 60 <= int(timeout) <= 1800:
        raise ValueError('Runtime deadline must be between 60 and 1800 seconds')
    rate = money(max_hourly_usd, 'Maximum hourly rate', positive=True)
    return {'status': 'approved-inputs', 'offer_id': int(offer_id),
            'max_hourly_usd': str(rate), 'test_timeout_seconds': int(timeout)}


def validate_offer(payload, offer_id, maximum, disk_gb, require_amd64=False,
                   max_estimated_usd='1.00', image_bytes=None):
    offers = payload.get('offers') if isinstance(payload, dict) else payload
    if not isinstance(offers, list):
        raise ValueError('Offer response is not an offers list')
    matches = [item for item in offers if isinstance(item, dict) and str(item.get('id')) == str(offer_id)]
    if len(matches) != 1:
        raise ValueError('The exact approved offer is missing or ambiguous; no replacement will be selected')
    offer = matches[0]
    if offer.get('rentable') is not True:
        raise ValueError('Approved offer is not currently rentable')
    for field in ('storage_cost', 'inet_down_cost', 'inet_up_cost'):
        money(offer.get(field), field)
    rate = money(offer.get('dph_total'), 'Quoted hourly rate')
    limit = money(maximum, 'Maximum hourly rate', positive=True)
    if rate > limit:
        raise ValueError(f'Quoted hourly rate {rate} exceeds approved maximum {limit}')
    if money(offer.get('disk_space'), 'Available disk') < disk_gb:
        raise ValueError('Offer has insufficient disk for the requested allocation')
    if require_amd64 and str(offer.get('cpu_arch', '')).lower() not in {'amd64', 'x86_64'}:
        raise ValueError('This application requires an explicitly identified x86-64 host')
    if require_amd64:
        # Deliberate baseline for this full-model smoke test, not a universal
        # minimum for every model or every GPU supported by the application.
        for field, minimum in (('num_gpus', 1), ('gpu_ram', 8000), ('cpu_ram', 8000), ('direct_port_count', 1)):
            if money(offer.get(field), field) < minimum:
                raise ValueError(f'Offer is below this test\'s {field} baseline {minimum}')
    budget = money(max_estimated_usd, 'Maximum estimated session cost', positive=True)
    if type(image_bytes) is not int or image_bytes <= 0:
        raise ValueError('Measured prepared-image size is required for the transfer estimate')
    # Use the entire 90-minute job ceiling, even though packaging precedes rent.
    # The pull allowance uses uncompressed image size plus 25% and 250MB for
    # provider SSH/bootstrap traffic. This is an estimate, not a billing guarantee.
    incoming_gb = Decimal(image_bytes) / Decimal(1_000_000_000) * Decimal('1.25') + Decimal('.25')
    outgoing_gb = Decimal('.05')
    estimated = (rate * Decimal('1.5') + incoming_gb * money(offer['inet_down_cost'], 'Inbound transfer price') +
                 outgoing_gb * money(offer['inet_up_cost'], 'Outbound transfer price'))
    if estimated > budget:
        raise ValueError(f'Conservative session/transfer estimate {estimated} exceeds approved maximum {budget}')
    # Keep account data and any unexpected fields out of the retained evidence.
    names = ('id', 'machine_id', 'gpu_name', 'num_gpus', 'gpu_ram', 'cpu_arch', 'cpu_ram',
             'dph_total', 'storage_cost', 'inet_down_cost', 'inet_up_cost',
             'disk_space', 'rentable', 'verified', 'verification', 'direct_port_count')
    return {'status': 'offer-accepted', 'offer': {name: offer[name] for name in names if name in offer},
            'requested_disk_gb': disk_gb, 'max_hourly_usd': str(limit),
            'max_estimated_usd': str(budget), 'conservative_estimate_usd': str(estimated),
            'estimate_inputs': {'job_hours': '1.5', 'image_bytes': image_bytes,
                                'incoming_gb': str(incoming_gb), 'outgoing_gb': str(outgoing_gb)},
            'quote_scope': 'On-demand search with --storage equal to --disk. Transfer allowance is included in the estimate; unconfirmed cleanup or provider anomalies can exceed it.'}


def instance_row(payload):
    row = payload.get('instances', payload) if isinstance(payload, dict) else None
    if not isinstance(row, dict):
        raise ValueError('Instance is missing or response shape is unsupported')
    return row


def validate_instance(payload, instance_id, maximum):
    row = instance_row(payload)
    if str(row.get('id')) != str(instance_id):
        raise ValueError('Status response belongs to a different instance')
    rate = money(row.get('dph_total'), 'Actual instance hourly rate')
    if rate > money(maximum, 'Maximum hourly rate', positive=True):
        raise ValueError('Actual instance hourly rate exceeds the approved maximum')
    return {'instance_id': int(instance_id), 'actual_status': row.get('actual_status', ''),
            'dph_total': str(rate)}


def success_response(payload):
    if not isinstance(payload, dict) or payload.get('success') is not True:
        raise ValueError('Vast did not confirm success; cleanup remains unconfirmed')


def docker_identity(image):
    result = subprocess.run(['docker', 'image', 'inspect', image], check=True,
                            text=True, stdout=subprocess.PIPE, timeout=30)
    records = json.loads(result.stdout)
    if len(records) != 1 or records[0]['Architecture'] != 'amd64' or records[0]['Os'] != 'linux':
        raise ValueError('Prepared image must be a single Linux amd64 image')
    return records[0]['Id']


def check_preflight_report(report, pin):
    required = {'status': 'passed', 'mode': 'preflight', 'model_executed': True,
                'hardware_verified': False, 'identities_unchanged': True,
                'source_commit': pin['source_commit'], 'application_sha256': pin['application']['sha256']}
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in required.items()):
        raise ValueError('A successful full-model software container preflight is required')
    if report.get('compiler_check', {}).get('found') != []:
        raise ValueError('Preflight did not establish known compiler absence')
    devices = report.get('devices', [])
    if (not isinstance(devices, list) or len(devices) != 2 or
            {d.get('provider') for d in devices} != {'embedded', 'native'} or
            any(d.get('software') is not True or d.get('hardware_required_rejection_verified') is not True for d in devices)):
        raise ValueError('Preflight did not verify both software providers and hardware-required rejection')
    image = report.get('image', {})
    if (not image.get('png') or image.get('execution', {}).get('provider') != 'native' or
            image.get('execution', {}).get('software') is not True):
        raise ValueError('Preflight lacks native software image-inference evidence')


def record_image(directory, image, report_path, commit, run_id):
    pin = json.loads((HERE / 'PIN.json').read_text())
    report = json.loads(report_path.read_text())
    check_preflight_report(report, pin)
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'[0-9]+', run_id):
        raise ValueError('Invalid workflow provenance')
    archive = directory / 'runtime.tar.gz'
    metadata = {'schema': 1, 'image': image, 'image_id': docker_identity(image),
                'archive': archive.name, 'archive_bytes': archive.stat().st_size,
                'archive_sha256': sha256(archive), 'preflight_report_sha256': sha256(report_path),
                'setup_commit': commit, 'workflow_run_id': run_id,
                'application_source_commit': pin['source_commit'],
                'application_sha256': pin['application']['sha256'],
                'scope': 'Software Docker preflight passed. Physical GPU and CUDA inference are not established.'}
    (directory / 'IMAGE.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (directory / 'preflight-report.json').write_bytes(report_path.read_bytes())
    return metadata


def check_image(directory, image, commit, run_id):
    metadata = json.loads((directory / 'IMAGE.json').read_text())
    pin = json.loads((HERE / 'PIN.json').read_text())
    expected = {'schema': 1, 'image': image, 'archive': 'runtime.tar.gz',
                'setup_commit': commit, 'workflow_run_id': run_id,
                'application_source_commit': pin['source_commit'],
                'application_sha256': pin['application']['sha256']}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Prepared image provenance does not match this workflow and pinned application')
    archive = directory / 'runtime.tar.gz'
    if archive.stat().st_size != metadata['archive_bytes'] or sha256(archive) != metadata['archive_sha256']:
        raise ValueError('Prepared Docker archive is corrupt')
    report = directory / 'preflight-report.json'
    if sha256(report) != metadata['preflight_report_sha256']:
        raise ValueError('Preflight report identity differs')
    report_data = json.loads(report.read_text())
    check_preflight_report(report_data, pin)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('approval')
    offer = sub.add_parser('check-offer')
    offer.add_argument('--input', required=True, type=Path)
    offer.add_argument('--offer-id', required=True)
    offer.add_argument('--maximum', required=True)
    offer.add_argument('--disk-gb', type=int, default=20)
    offer.add_argument('--require-amd64', action='store_true')
    offer.add_argument('--max-estimated-usd', required=True)
    offer.add_argument('--image-metadata', required=True, type=Path)
    instance = sub.add_parser('check-instance')
    instance.add_argument('--input', required=True, type=Path)
    instance.add_argument('--instance-id', required=True)
    instance.add_argument('--maximum', required=True)
    instance.add_argument('--status-only', action='store_true')
    destroy = sub.add_parser('check-success')
    destroy.add_argument('--input', required=True, type=Path)
    destroy.add_argument('--python-repr', action='store_true',
                         help='Normalize the pinned attach-ssh command\'s Python literal; never evaluate code')
    for name in ('record-image', 'check-image'):
        image = sub.add_parser(name)
        image.add_argument('--directory', required=True, type=Path)
        image.add_argument('--image', required=True)
        image.add_argument('--commit', required=True)
        image.add_argument('--run-id', required=True)
        if name == 'record-image':
            image.add_argument('--preflight-report', required=True, type=Path)
        else:
            image.add_argument('--loaded', action='store_true')
    args = parser.parse_args()
    if args.command == 'approval':
        money(os.environ['MAX_ESTIMATED_USD'], 'Maximum estimated session cost', positive=True)
        result = approval_gate(*(os.environ[name] for name in (
            'APPROVAL', 'OFFER_ID', 'BASE_IMAGE', 'MAX_HOURLY_USD', 'BACKEND',
            'DRIVER_CAPABILITIES', 'TEST_TIMEOUT_SECONDS')))
    elif args.command == 'check-offer':
        images = json.loads(args.image_metadata.read_text())
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError('Expected exactly one prepared image metadata record')
        result = validate_offer(json.loads(args.input.read_text()), args.offer_id,
                                args.maximum, args.disk_gb, args.require_amd64,
                                args.max_estimated_usd, images[0].get('Size'))
    elif args.command == 'check-instance':
        result = validate_instance(json.loads(args.input.read_text()), args.instance_id, args.maximum)
        if args.status_only:
            print(result['actual_status'])
            return
    elif args.command == 'check-success':
        content = args.input.read_text()
        if len(content) > 65536:
            raise ValueError('Unexpectedly large CLI response')
        success_response(ast.literal_eval(content) if args.python_repr else json.loads(content))
        result = {'success': True}
    elif args.command == 'record-image':
        result = record_image(args.directory, args.image, args.preflight_report, args.commit, args.run_id)
    else:
        result = check_image(args.directory, args.image, args.commit, args.run_id)
        if args.loaded and docker_identity(args.image) != result['image_id']:
            raise ValueError('Loaded Docker image ID differs from the tested image')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
