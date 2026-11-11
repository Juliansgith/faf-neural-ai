try:
    import no_network
except ImportError:
    from tests import no_network
import unittest
from pathlib import Path
from unittest.mock import patch
from fa_train import autopilot


class ShutdownTests(unittest.TestCase):
    def pilot(self):
        p = object.__new__(autopilot.Autopilot)
        p.dry, p.state, p.s = False, {'mode': 'running'}, {}
        p.run = Path(__file__).resolve().parent / 'managed_run'
        p.loop_procs = lambda procs: [v for v in procs if 'fa_train.rl_loop' in v['cmd']]
        return p

    def test_command_requests_hold_and_drain(self):
        p = self.pilot()
        with patch.object(autopilot, 'log_event'):
            p.apply_command({'cmd': 'shutdown'}, 'rl_188')
        self.assertEqual(p.state, {'mode': 'hold', 'shutdown_requested': True})
        self.assertTrue(p.shutdown_ready([]))

    def test_any_active_training_or_unknown_state_defers_shutdown(self):
        p = self.pilot()
        p.state = {'mode': 'hold', 'shutdown_requested': True}
        for procs in (None, [{'cmd': 'python -m fa_train.rl_loop'}],
                      [{'cmd': 'python -m fa_train.learn_rl'}]):
            with self.subTest(procs=procs):
                self.assertFalse(p.shutdown_ready(procs))
        self.assertTrue(p.shutdown_ready([{'cmd': 'python -m fa_train.player'}]))

    def test_requires_explicit_command_hold_and_non_dry_mode(self):
        p = self.pilot()
        self.assertFalse(p.shutdown_ready([]))
        p.state['shutdown_requested'] = True
        self.assertFalse(p.shutdown_ready([]))
        p.state['mode'] = 'hold'
        p.dry = True
        self.assertFalse(p.shutdown_ready([]))

    def test_resume_cancels_a_pending_shutdown(self):
        p = self.pilot()
        with patch.object(autopilot, 'log_event'):
            p.apply_command({'cmd': 'shutdown'}, 'rl_188')
            p.apply_command({'cmd': 'resume'}, 'rl_188')
        self.assertNotIn('shutdown_requested', p.state)
        self.assertEqual(p.state['mode'], 'running')


if __name__ == '__main__':
    unittest.main()
