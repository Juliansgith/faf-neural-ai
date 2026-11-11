import copy
from collections import Counter
import unittest

from fa_train.benchmark import game_key, schedule_hash, suite, summarize, validate_rows


class BenchmarkTests(unittest.TestCase):
    def test_full_cross_and_published_weights(self):
        rows = suite(1730010000)
        self.assertEqual(len(rows), 640)
        self.assertEqual(len({game_key(r) for r in rows}), 640)
        self.assertEqual(Counter(r['map'] for r in rows)['SCMP_012'], 128)
        self.assertEqual(Counter(r['map'] for r in rows)['SCMP_023'], 64)
        self.assertEqual(len({(r['map'], r['faction'], r['ai_faction'], tuple(r['slots'])) for r in rows}), 224)
        self.assertEqual(rows, suite(1730010000))
        self.assertNotEqual(schedule_hash(rows), schedule_hash(suite(1730020000)))

    def test_complete_and_confident_required(self):
        rows = suite(20)
        results = [dict(r, outcome='victory' if i < 420 else 'timeout') for i, r in enumerate(rows)]
        summary = summarize(rows, results)
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['statistical_threshold_met'])
        self.assertEqual(summary['total']['games'], 640)
        self.assertEqual(summary['total']['outcomes']['timeout'], 220)
        self.assertFalse(summarize(rows, results[:-1])['statistical_threshold_met'])
        borderline = [dict(r, outcome='victory' if i < 385 else 'defeat') for i, r in enumerate(rows)]
        self.assertGreater(summarize(rows, borderline)['total']['rate'], .60)
        self.assertFalse(summarize(rows, borderline)['statistical_threshold_met'])

    def test_failures_and_duplicates_never_disappear(self):
        rows = suite(20)
        results = [dict(r, outcome='victory') for r in rows]
        for changed in (results + results[:1], results + [{}],
                        results[:-1] + [dict(results[-1], outcome='error')],
                        results[:-1] + [dict(results[-1], seed=999999)]):
            self.assertFalse(summarize(rows, changed)['complete'])
            self.assertFalse(summarize(rows, changed)['statistical_threshold_met'])

    def test_rejects_unsafe_or_ambiguous_schedule(self):
        row = suite(20)[0]
        for field, value in [('minutes', float('nan')), ('minutes', -1), ('seed', 2**31),
                             ('faction', 1.5), ('slots', [1, 1]), ('resource_gift', 1000)]:
            bad = copy.deepcopy(row)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_rows([bad])
        with self.assertRaises(ValueError):
            validate_rows([row, row])


if __name__ == '__main__':
    unittest.main()
