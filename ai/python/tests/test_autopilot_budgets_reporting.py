"""Budget and reporting regressions from the October 25 live-run audit; no processes or network."""
try:
    import no_network
except ImportError:
    from tests import no_network
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from fa_train import autopilot as ap


class Budgets(unittest.TestCase):
    def test_productive_retry_stops_at_its_30_hour_budget(self):
        exp = ap.next_experiment([{'name': 'X3'}], {'X3': {'requeue': True}}, 'hold')
        rules = ap.experiment_rules(ap.DEFAULTS['rules'], exp)
        verdict = ap.judge(rules, {}, 'champ', [], 0, 31, 31)
        self.assertEqual(verdict.kind, 'MAX')
        self.assertEqual(verdict.numbers['max_hours'], 30)
        self.assertEqual(ap.DEFAULTS['rules']['max_hours'], 60)

    def test_explicit_12_hour_budget_does_not_inherit_24_hour_grace(self):
        rules = ap.experiment_rules(ap.DEFAULTS['rules'], {'max_hours': 12})
        self.assertEqual(ap.judge(rules, {}, 'champ', [], 0, 12, 1).kind, 'MAX')
        self.assertEqual(ap.judge(rules, {}, 'champ', [], 0, 12, 1, extra_hours=2).kind, 'continue')

    def test_default_budget_keeps_promotion_grace(self):
        rules = ap.experiment_rules(ap.DEFAULTS['rules'], {})
        self.assertEqual(ap.judge(rules, {}, 'champ', [], 0, 61, 1).kind, 'continue')

    def test_invalid_budget_rejected(self):
        for value in [-1, 0, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                ap.experiment_rules(ap.DEFAULTS['rules'], {'max_hours': value})

    def test_selfplay_keeps_successful_update_settings(self):
        exp = ap.expand(next(e for e in ap.DEFAULT_QUEUE if e['name'] == 'X6-self-play'), ap.DEFAULT_QUEUE)
        self.assertEqual(exp['learn']['--lr'], '1e-5')
        self.assertEqual(exp['learn']['--epochs'], '2')
        self.assertEqual(exp['loop']['--games-per-iteration'], '720')
        self.assertEqual(exp['loop']['--league'], 'CHAMPION')


class Reporting(unittest.TestCase):
    def test_report_preserves_current_verdict_evidence(self):
        pilot = ap.Autopilot.__new__(ap.Autopilot)
        pilot.s = {}; pilot.scan = {'champ': {}}; pilot.run = Path('unused')
        numbers = {'pooled': [0.01, 0.02, 400], 'fallbacks': 1, 'max_hours': 30}
        pilot.state = {'verdict': {'kind': 'continue', 'why': '', 'numbers': numbers}}
        with mock.patch.object(ap, 'read_journal', return_value=[]), mock.patch.object(ap, 'write_report') as write:
            pilot.write('champ')
        self.assertEqual(write.call_args.args[-1].numbers, numbers)

    def test_completed_run_keeps_its_own_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot = ap.Autopilot.__new__(ap.Autopilot)
            pilot.run = Path(temp); pilot.s = {}; pilot.dry = True; pilot.alerts = mock.Mock()
            pilot.state = {'results': {'X3': {'promotions': 4}}, 'experiment': {
                'name': 'X3', 'started': '2026-10-25T00:00:00', 'start_row': 1,
                'run_promotions': 0, 'spec': {'_requeue': True}}}
            rows = [{'checkpoint': 'rl_001.pt'}, {'checkpoint': 'rl_002.pt'},
                    {'checkpoint': 'rl_188.pt', 'fallback': True, 'gate': 'bad'}]
            with mock.patch.object(ap, 'read_journal', return_value=rows), mock.patch.object(ap, 'log_event'):
                pilot.end_experiment(ap.Verdict('MAX', 'budget', {'networks': 1, 'fallbacks': 1}), [])
            run = pilot.state['runs'][0]
            self.assertEqual((run['trained'], run['promotions'], run['fallbacks']), (1, 0, 1))
            self.assertEqual(pilot.state['results']['X3']['promotions'], 4)
            self.assertTrue(pilot.state['results']['X3']['requeued_done'])


if __name__ == '__main__':
    unittest.main()
