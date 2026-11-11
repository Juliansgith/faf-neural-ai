"""Independent CPU experiments must not look like orphaned GPU training."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fa_train import autopilot as ap


class LearnerScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'production'
        self.output = self.root / 'separate trial' / 'candidate.pt'

    def learner(self, *args, pid=2, ppid=0):
        return {'pid': pid, 'ppid': ppid, 'cmd': subprocess.list2cmdline(
            ['python', '-m', 'fa_train.learn_rl', *args])}

    def test_explicit_cpu_with_quoted_external_output_and_readonly_reference(self):
        for args in [('--device', 'cpu'), ('--device=cpu',)]:
            row = self.learner(*args, '--out', str(self.output),
                               '--init', str(self.run / 'checkpoints' / 'rl_188.pt'))
            self.assertEqual(ap.learner_processes([row], self.run), [])

    def test_gpu_default_ambiguous_and_nonabsolute_jobs_remain_blocking(self):
        variants = [[], ['--device', 'cuda'], ['--device', 'cpu', '--device', 'cuda'],
                    ['--device', 'cpu', '--device', 'cpu'], ['--device=cpu:0']]
        for args in variants:
            row = self.learner(*args, '--out', str(self.output))
            self.assertEqual(ap.learner_processes([row], self.run), [row])
        for output in [None, 'relative/candidate.pt', str(self.run / 'candidate.pt')]:
            row = self.learner('--device', 'cpu', *(['--out', output] if output else []))
            self.assertEqual(ap.learner_processes([row], self.run), [row])

    def test_shared_mutable_paths_remain_blocking(self):
        for flag in ('--mapped', '--stop-file'):
            row = self.learner('--device', 'cpu', '--out', str(self.output), flag, str(self.run / 'shared'))
            self.assertEqual(ap.learner_processes([row], self.run), [row])

    def test_flags_inside_inline_code_are_not_device_arguments(self):
        row = {'pid': 2, 'cmd': subprocess.list2cmdline(['python', '-c',
            "print('--device cpu');import fa_train.learn_rl", '--out', str(self.output)])}
        self.assertEqual(ap.learner_processes([row], self.run), [row])

    def test_managed_child_through_launcher_remains_blocking(self):
        loop = {'pid': 1, 'ppid': 0, 'cmd': f'python -m fa_train.rl_loop --run "{self.run}"'}
        launcher = {'pid': 5, 'ppid': 1, 'cmd': 'python launcher.py'}
        row = self.learner('--device', 'cpu', '--out', str(self.output), ppid=5)
        self.assertEqual(ap.learner_processes([loop, launcher, row], self.run), [row])

    def test_parser_failure_stays_blocking(self):
        row = self.learner('--device', 'cpu', '--out', str(self.output))
        with patch.object(ap, '_command_args', return_value=None):
            self.assertEqual(ap.learner_processes([row], self.run), [row])

    def test_shutdown_can_drain_while_independent_cpu_work_continues(self):
        pilot = object.__new__(ap.Autopilot)
        pilot.run, pilot.dry = self.run, False
        pilot.state = {'mode': 'hold', 'shutdown_requested': True}
        row = self.learner('--device', 'cpu', '--out', str(self.output))
        self.assertTrue(pilot.shutdown_ready([row]))
        self.assertFalse(pilot.shutdown_ready([self.learner('--out', str(self.output))]))


if __name__ == '__main__':
    unittest.main()
