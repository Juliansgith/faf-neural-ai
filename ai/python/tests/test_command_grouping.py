import unittest
from collections import defaultdict
from fa_env.command_grouping import group_independent


def per_unit(commands):
    result = defaultdict(list)
    for command in commands:
        for subject in command["subject"]:
            result[subject].append({k: v for k, v in command.items() if k != "subject"})
    return dict(result)


class IndependentGroupingTest(unittest.TestCase):
    def check(self, commands, expected_count):
        actual = group_independent(commands, {"move", "attack"}, {"build": 0, "move": 1, "attack": 2})
        self.assertEqual(per_unit(actual), per_unit(commands))
        self.assertEqual(len(actual), expected_count)
        return actual

    def test_interleaved_independent_destinations_fit_fewer_slots(self):
        commands = [{"verb": "move", "subject": [str(i)], "cell": (i % 2, 4)} for i in range(20)]
        actual = self.check(commands, 2)
        self.assertEqual(len(actual[0]["subject"]), 10)
        self.assertEqual(commands[0]["subject"], ["0"])

    def test_dependent_queue_barrier_and_repeated_orders_survive(self):
        self.check([{"verb": "move", "subject": ["a"], "cell": (1, 2)},
                    {"verb": "attack", "subject": ["a"], "target": "enemy"},
                    {"verb": "move", "subject": ["a", "b"], "cell": (1, 2)}], 3)
        self.check([{"verb": "move", "subject": ["a"], "cell": (1, 2)}] * 2, 2)

    def test_single_subject_orders_and_carry_provenance_are_not_merged(self):
        self.check([{"verb": "build", "subject": [u], "bp": "pgen"} for u in ["a", "b"]], 2)
        self.check([{"verb": "move", "subject": ["a"], "cell": (1, 2)},
                    {"verb": "move", "subject": ["b"], "cell": (1, 2), "carried": True}], 2)

    def test_many_queue_interleavings_preserve_every_units_exact_commands(self):
        import random
        rng = random.Random(47)
        for _ in range(100):
            commands = [{"verb": rng.choice(["move", "attack", "build"]),
                         "subject": rng.sample(list("abcdef"), rng.randrange(1, 3)),
                         "argument": rng.randrange(3)} for _ in range(35)]
            actual = group_independent(commands, {"move", "attack"}, {"build": 0, "move": 1, "attack": 2})
            self.assertEqual(per_unit(actual), per_unit(commands))
            self.assertLessEqual(len(actual), len(commands))


if __name__ == "__main__":
    unittest.main()
