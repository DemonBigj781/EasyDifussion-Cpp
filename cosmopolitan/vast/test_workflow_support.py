"""Exercise paid-workflow failure gates entirely offline; no cloud SDK imported."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workflow_support as support


class ApprovalTests(unittest.TestCase):
    def gate(self, **changes):
        values = dict(approval='APPROVE', offer_id='42', base_image='edcpp-cosmo-runtime:1-1',
                      max_hourly_usd='0.10', backend='cosmo-vulkan-x86_64',
                      capabilities='compute,utility,graphics', timeout=660)
        values.update(changes)
        return support.approval_gate(**values)

    def test_approval_is_specific_and_numeric(self):
        self.assertEqual(self.gate()['offer_id'], 42)
        for value in ('', 'approve', 'APPROVE\n', 'yes'):
            with self.subTest(approval=value), self.assertRaises(ValueError):
                self.gate(approval=value)
        for value in ('', '42; echo unexpected', '42\n', '-42', 'cheapest'):
            with self.subTest(offer_id=value), self.assertRaises(ValueError):
                self.gate(offer_id=value)

    def test_malformed_or_unbounded_inputs_fail(self):
        for key, value in [('max_hourly_usd', 'NaN'), ('max_hourly_usd', 'Infinity'),
                           ('max_hourly_usd', '-0.1'), ('max_hourly_usd', '0'),
                           ('base_image', 'ubuntu:24.04\nRUN false'),
                           ('backend', 'name/invalid'), ('capabilities', 'all'),
                           ('timeout', 0), ('timeout', 3600)]:
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                self.gate(**{key: value})


class OfferTests(unittest.TestCase):
    def setUp(self):
        self.offer = {'id': 42, 'cpu_arch': 'amd64', 'rentable': True,
                      'disk_space': 100, 'dph_total': 0.08,
                      'storage_cost': 0.15, 'inet_down_cost': 0.01,
                      'inet_up_cost': 0.02, 'gpu_name': 'fixture GPU',
                      'num_gpus': 1, 'gpu_ram': 8192, 'cpu_ram': 16000, 'direct_port_count': 1,
                      'unexpected_account_field': 'must not be retained'}

    def check(self, offers=None, limit='0.10', budget='1.00'):
        return support.validate_offer([self.offer] if offers is None else offers, '42', limit, 20, True,
                                      budget, 2_000_000_000)

    def test_exact_offer_and_boundary_price_pass(self):
        result = self.check()
        self.assertEqual(result['offer']['id'], 42)
        self.assertNotIn('unexpected_account_field', result['offer'])
        self.offer['dph_total'] = '0.10'
        self.check()
        self.offer['dph_total'] = '3'
        with self.assertRaises(ValueError):
            self.check()
        with self.assertRaises(ValueError):
            self.check(limit='3')
        self.check(limit='3', budget='5')

    def test_transfer_prices_cannot_bypass_the_session_budget(self):
        self.offer['inet_down_cost'] = '10'
        with self.assertRaisesRegex(ValueError, 'session/transfer estimate'):
            self.check()

    def test_price_and_hardware_fail_closed(self):
        for key, value in [('dph_total', 0.10001), ('dph_total', None),
                           ('dph_total', True), ('dph_total', 'NaN'),
                           ('dph_total', 'Infinity'), ('dph_total', -1),
                           ('cpu_arch', 'aarch64'), ('cpu_arch', None),
                           ('storage_cost', 'NaN'), ('inet_down_cost', None),
                           ('inet_up_cost', -1), ('gpu_ram', 4000), ('cpu_ram', 4000),
                           ('direct_port_count', 0), ('num_gpus', 0),
                           ('disk_space', 19), ('rentable', False)]:
            bad = copy.deepcopy(self.offer)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check([bad])

    def test_missing_duplicate_and_wrong_offer_never_choose_a_replacement(self):
        other = {**self.offer, 'id': 99, 'dph_total': 0.01}
        for payload in ([], [other], [self.offer, self.offer], {}, {'offers': None}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.check(payload)
        self.assertEqual(self.check({'offers': [other, self.offer]})['offer']['id'], 42)

    def test_instance_rate_rechecked_and_cleanup_never_false_green(self):
        row = {'id': 7, 'actual_status': 'running', 'dph_total': 0.09}
        self.assertEqual(support.validate_instance(row, '7', '.10')['actual_status'], 'running')
        self.assertEqual(support.validate_instance({'instances': row}, '7', '.10')['instance_id'], 7)
        for payload in ({'instances': None}, {**row, 'id': 8}, {**row, 'dph_total': .11}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                support.validate_instance(payload, '7', '.10')
        support.success_response({'success': True})
        for payload in ({}, {'success': False}, {'success': 'true'}, {'success': 1}, None):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                support.success_response(payload)


class ImageIdentityTests(unittest.TestCase):
    def report(self):
        pin = json.loads((support.HERE / 'PIN.json').read_text())
        return {'status': 'passed', 'mode': 'preflight', 'model_executed': True,
                'hardware_verified': False, 'identities_unchanged': True,
                'source_commit': pin['source_commit'], 'application_sha256': pin['application']['sha256'],
                'compiler_check': {'found': []},
                'devices': [{'provider': p, 'software': True, 'hardware_required_rejection_verified': True}
                            for p in ('embedded', 'native')],
                'image': {'png': {'fixture': True}, 'execution': {'provider': 'native', 'software': True}}}

    def test_corrupt_archive_report_or_wrong_run_cannot_reach_paid_packaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / 'runtime.tar.gz'
            archive.write_bytes(b'fixture archive; never loaded into Docker')
            report = directory / 'report.json'
            report.write_text(json.dumps(self.report()))
            with patch.object(support, 'docker_identity', return_value='sha256:' + '1' * 64):
                support.record_image(directory, 'example:1', report, 'a' * 40, '12')
            support.check_image(directory, 'example:1', 'a' * 40, '12')
            with self.assertRaises(ValueError):
                support.check_image(directory, 'example:1', 'a' * 40, '13')
            with self.assertRaises(ValueError):
                support.check_image(directory, 'example:1', 'b' * 40, '12')
            archive.write_bytes(b'corrupt archive')
            with self.assertRaises(ValueError):
                support.check_image(directory, 'example:1', 'a' * 40, '12')
            archive.write_bytes(b'fixture archive; never loaded into Docker')
            (directory / 'preflight-report.json').write_text('{"status":"passed"}')
            with self.assertRaises(ValueError):
                support.check_image(directory, 'example:1', 'a' * 40, '12')

    def test_failed_or_hardware_report_does_not_substitute_for_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for mode, status in [('preflight', 'failed'), ('hardware', 'passed'), ('package', 'passed')]:
                report = directory / 'report.json'
                report.write_text(json.dumps({'status': status, 'mode': mode}))
                with self.subTest(mode=mode, status=status), self.assertRaises(ValueError):
                    support.record_image(directory, 'example:1', report, 'a' * 40, '12')

    def test_missing_model_or_provider_proof_cannot_be_exported(self):
        pin = json.loads((support.HERE / 'PIN.json').read_text())
        for field, value in [('model_executed', False), ('identities_unchanged', False),
                             ('hardware_verified', True), ('devices', []),
                             ('compiler_check', {'found': ['cc']}), ('image', {}),
                             ('application_sha256', '0' * 64)]:
            report = self.report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                support.check_preflight_report(report, pin)


if __name__ == '__main__':
    unittest.main()
