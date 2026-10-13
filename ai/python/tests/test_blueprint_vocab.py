import unittest
from fa_env.blueprint_vocab import blueprint_indices, BLUEPRINT_INDEX


class BlueprintVocabularyTests(unittest.TestCase):
    def test_teacher_mod_entries_cannot_shift_common_units(self):
        ordinary = {'uel0105': {}, 'uel0201': {}, 'ueb0101': {}}
        modded = {**ordinary, 'mai2806': {}, 'aaa_future_mod': {}}
        a, b = blueprint_indices(ordinary), blueprint_indices(modded)
        self.assertEqual(a, {k:b[k] for k in ordinary})
        self.assertEqual(a['uel0201'], 317)
        self.assertEqual(a['uel0105'], 314)
        self.assertEqual(b['aaa_future_mod'], 0)
        self.assertGreater(b['mai2806'], 743)

    def test_checkpoint_embedding_capacity_and_unique_ids(self):
        self.assertEqual(len(BLUEPRINT_INDEX), len(set(BLUEPRINT_INDEX.values())))
        self.assertLess(max(BLUEPRINT_INDEX.values()), 1024)


if __name__ == '__main__':
    unittest.main()
