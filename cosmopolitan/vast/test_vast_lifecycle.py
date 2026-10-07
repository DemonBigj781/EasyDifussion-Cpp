#!/usr/bin/env python3
"""Offline failure tests. Fake CLI only: no Vast API or credentials are used."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vast_lifecycle import Cli, Lifecycle, LifecycleError, rows_from


def result(body=None, code=0, *, raw=None, timeout=False):
    return {'returncode': code, 'timed_out': timeout,
            'stdout': json.dumps(body) if raw is None else raw}


def row(value=77, label='test-run-1'):
    return {'id': value, 'label': label, 'actual_status': 'running'}


class Clock:
    def __init__(self):
        self.now = 0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class FakeCli:
    def __init__(self, *, listings=None, creation=None, destroys=None, singles=None, version='vastai 1.8.3'):
        self.listings = list(listings or [result([])])
        self.creation = creation or result({'success': True, 'new_contract': 77})
        self.destroys = list(destroys or [result(raw='')])
        self.singles = list(singles or [result({'instances': None})])
        self.version = version
        self.calls = []
    def call(self, args, timeout):
        assert 0 < timeout <= 35
        self.calls.append(list(args))
        if args == ['--version']:
            return result(raw=self.version)
        if args[:2] == ['show', 'instances']:
            assert args[2:] == ['--all', '--raw', '--retry', '1']
            return self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]
        if args[:2] == ['show', 'instance']:
            assert args[-3:] == ['--raw', '--retry', '1']
            return self.singles.pop(0) if len(self.singles) > 1 else self.singles[0]
        if args[:2] == ['create', 'instance']:
            assert args[-2:] == ['--retry', '1']
            assert '--cancel-unavail' in args
            return self.creation
        if args[:2] == ['destroy', 'instance']:
            assert args[-4:] == ['-y', '--raw', '--retry', '1']
            return self.destroys.pop(0) if len(self.destroys) > 1 else self.destroys[0]
        raise AssertionError(args)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.clock = Clock()
        self.output = self.directory / 'github-output'
    def tearDown(self):
        self.temp.cleanup()
    def make(self, cli):
        return Lifecycle(cli, self.directory, github_output=self.output,
                         clock=self.clock, sleep=self.clock.sleep, poll_seconds=1)
    def create(self, lifecycle):
        return lifecycle.create(offer_id=42, image='ghcr.io/test/runtime@sha256:' + 'a'*64,
                                label='test-run-1', capabilities='compute,utility,graphics',
                                reconcile_timeout=3)
    def record(self, name):
        return json.loads((self.directory / name).read_text())
    def test_success_id_is_persisted(self):
        cli = FakeCli()
        actual = self.create(self.make(cli))
        self.assertEqual(actual['instance_id'], 77)
        self.assertEqual(actual['status'], 'created')
        self.assertIn('instance_id=77', self.output.read_text())
        self.assertEqual(sum(x[:2] == ['create','instance'] for x in cli.calls), 1)
    def test_preexisting_label_refuses_creation(self):
        cli = FakeCli(listings=[result([row()])])
        with self.assertRaises(LifecycleError): self.create(self.make(cli))
        self.assertFalse(any(x[:2] == ['create','instance'] for x in cli.calls))
    def test_version_mismatch_refuses_any_api(self):
        cli = FakeCli(version='vastai 1.8.4')
        with self.assertRaises(LifecycleError): self.create(self.make(cli))
        self.assertEqual(cli.calls, [['--version']])
    def test_listing_error_does_not_create(self):
        cli = FakeCli(listings=[result([], code=1)])
        with self.assertRaises(LifecycleError): self.create(self.make(cli))
        self.assertFalse(any(x[:2] == ['create','instance'] for x in cli.calls))
    def test_timeout_recovers_exact_unique_label_without_retry(self):
        cli = FakeCli(listings=[result([]), result([]), result([row(88, 'other'), row()])],
                      creation=result(raw='', code=-9, timeout=True))
        actual = self.create(self.make(cli))
        self.assertEqual(actual['status'], 'created-response-reconciled')
        self.assertEqual(actual['instance_id'], 77)
        self.assertNotIn('other', (self.directory / 'creation.json').read_text())
        self.assertEqual(sum(x[:2] == ['create','instance'] for x in cli.calls), 1)
    def test_malformed_response_recovers(self):
        cli = FakeCli(listings=[result([]), result([row()])], creation=result(raw='bad json'))
        self.assertEqual(self.create(self.make(cli))['instance_id'], 77)
    def test_failed_success_with_id_preserves_for_cleanup(self):
        cli = FakeCli(creation=result({'success': False, 'new_contract': 77}))
        with self.assertRaises(LifecycleError): self.create(self.make(cli))
        self.assertEqual(self.record('creation.json')['instance_id'], 77)
        self.assertIn('instance_id=77', self.output.read_text())
    def test_no_matching_label_is_unconfirmed_not_success(self):
        cli = FakeCli(creation=result({'success': False}))
        with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'): self.create(self.make(cli))
        self.assertEqual(self.record('creation.json')['status'], 'creation-UNCONFIRMED')
        self.assertEqual(sum(x[:2] == ['create','instance'] for x in cli.calls), 1)
    def test_duplicate_label_is_unconfirmed(self):
        cli = FakeCli(listings=[result([]), result([row(77), row(78)])], creation=result(raw=''))
        with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'): self.create(self.make(cli))
        self.assertNotIn('instance_id=', self.output.read_text())
    def test_boolean_id_is_not_an_id(self):
        cli = FakeCli(creation=result({'success': True, 'new_contract': True}))
        with self.assertRaises(LifecycleError): self.create(self.make(cli))
        self.assertNotIn('instance_id=', self.output.read_text())
    def test_destroy_empty_output_needs_absence(self):
        cli = FakeCli(listings=[result([row()]), result([])])
        actual = self.make(cli).destroy(instance_id=77, timeout=3)
        self.assertEqual(actual['status'], 'destroyed-absence-confirmed')
        self.assertEqual(len(actual['events']), 4)
        self.assertTrue(actual['events'][-1]['explicit_absence'])
    def test_destroy_failure_but_absence_is_proof(self):
        cli = FakeCli(destroys=[result(raw='', code=1)], listings=[result([])])
        self.assertEqual(self.make(cli).destroy(instance_id=77, timeout=3)['status'],
                         'destroyed-absence-confirmed')
    def test_destroy_false_or_zero_exit_does_not_pass_while_present(self):
        cli = FakeCli(destroys=[result({'success': False})], listings=[result([row()])])
        with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'):
            self.make(cli).destroy(instance_id=77, timeout=3)
        self.assertEqual(self.record('cleanup.json')['status'], 'UNCONFIRMED')
    def test_empty_list_with_present_exact_id_is_not_deletion(self):
        cli = FakeCli(listings=[result([])], singles=[result(row())])
        with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'):
            self.make(cli).destroy(instance_id=77, timeout=3)
    def test_exact_query_http_error_is_not_absence(self):
        cli = FakeCli(listings=[result([])], singles=[result({'instances': None}, code=1)])
        with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'):
            self.make(cli).destroy(instance_id=77, timeout=3)
    def test_exact_query_requires_explicit_null_shape(self):
        for payload in ({}, [], {'success': True}, {'instances': []}, {'instances': {}},
                        {'instances': None, 'error': 'invalid'}):
            with self.subTest(payload=payload):
                cli = FakeCli(listings=[result([])], singles=[result(payload)])
                with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'):
                    self.make(cli).destroy(instance_id=77, timeout=2)
    def test_exact_query_invalid_json_does_not_confirm(self):
        cli = FakeCli(listings=[result([])], singles=[result(raw='not JSON')])
        with self.assertRaisesRegex(LifecycleError, 'UNCONFIRMED'):
            self.make(cli).destroy(instance_id=77, timeout=2)

    def test_failed_empty_listing_is_not_absence(self):
        cli = FakeCli(listings=[result([], code=1)])
        with self.assertRaises(LifecycleError): self.make(cli).destroy(instance_id=77, timeout=3)
    def test_unsupported_or_partial_listing_is_not_absence(self):
        for payload in ({'instances': []}, {}, [{'label':'x'}], [{'id':8}],
                        [row(9), row(9)], [row(True)]):
            with self.subTest(payload=payload), self.assertRaises(LifecycleError):
                rows_from(result(payload))
    def test_no_create_record_no_label_cleanup(self):
        with self.assertRaises(FileNotFoundError):
            self.make(FakeCli()).destroy(label='test-run-1', timeout=3)
    def test_interrupted_output_recovered_from_creation_record(self):
        life = self.make(FakeCli())
        self.create(life)
        self.output.unlink()
        cli = FakeCli(listings=[result([])])
        actual = self.make(cli).destroy(label='test-run-1', timeout=3)
        self.assertEqual(actual['instance_id'], 77)
    def test_ambiguous_create_late_id_recovered_for_destroy(self):
        cli = FakeCli(creation=result(raw=''))
        with self.assertRaises(LifecycleError): self.create(self.make(cli))
        cli = FakeCli(listings=[result([row()]), result([])])
        self.assertEqual(self.make(cli).destroy(label='test-run-1', timeout=3)['instance_id'], 77)
    def test_ambiguous_create_no_id_cleanup_stays_unconfirmed(self):
        with self.assertRaises(LifecycleError): self.create(self.make(FakeCli(creation=result(raw=''))))
        with self.assertRaisesRegex(LifecycleError, 'Unknown creation outcome'):
            self.make(FakeCli()).destroy(label='test-run-1', timeout=3)
        self.assertEqual(self.record('cleanup.json')['status'], 'UNCONFIRMED')
    def test_real_subprocess_fake_cli_timeout_and_secret_isolation(self):
        fake = self.directory / 'fake-vastai'
        fake.write_text('#!/usr/bin/env python3\nimport os,sys,time\n'
                        'assert os.environ["VAST_API_KEY"] == "offline-test-key"\n'
                        'assert "VASTAI_KEY" not in os.environ\n'
                        'assert "offline-test-key" not in sys.argv\n'
                        'if "wait" in sys.argv: time.sleep(5)\n'
                        'print("vastai 1.8.3")\n')
        fake.chmod(0o755)
        with patch.dict(os.environ, {'VASTAI_KEY':'offline-test-key'}):
            cli = Cli(str(fake))
            self.assertEqual(cli.call(['--version'], 2)['stdout'].strip(), 'vastai 1.8.3')
            actual = cli.call(['wait'], 0.05)
            self.assertTrue(actual['timed_out'])
            self.assertEqual(actual['returncode'], -9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
