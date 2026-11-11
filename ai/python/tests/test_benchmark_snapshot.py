from pathlib import Path
import hashlib
import tempfile
import unittest

from fa_train.local_benchmark import freeze_runtime
from fa_train.benchmark import lua_tree_hash


class SnapshotTests(unittest.TestCase):
    def test_mixed_case_lua_names_have_portable_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected = hashlib.sha256()
            # Explicit bytewise order: WindowsPath's default ordering puts
            # alpha.lua first, causing identical opponents to look different.
            for name in ('Zed.lua', 'alpha.lua', 'z.lua'):
                data = b'return true\n'
                (root / name).write_bytes(data)
                encoded = name.encode()
                expected.update(len(encoded).to_bytes(8, 'big') + encoded
                                + len(data).to_bytes(8, 'big') + data)
            self.assertEqual(lua_tree_hash(root), expected.hexdigest())

    def test_runtime_and_lua_are_independent_of_working_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ai, mod = root / 'source/ai', root / 'source/M28AI'
            for path, text in ((ai / 'python/fa_env/model.py', 'original python'),
                               (ai / 'python/fa_env/blueprint_vocab.json', '{}'),
                               (ai / 'agentbridge.lua', 'original bridge'),
                               (ai / 'fa_overrides/schook/test.lua', 'original hook'),
                               (ai / 'python/__pycache__/test.pyc', 'cached'),
                               (mod / 'mod_info.lua', 'version = 324'),
                               (mod / 'lua/test.lua', 'original opponent')):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            python, frozen_mod = freeze_runtime(ai, mod, root / 'snapshot')
            (ai / 'python/fa_env/model.py').write_text('changed python')
            (ai / 'fa_overrides/schook/test.lua').write_text('changed hook')
            (mod / 'lua/test.lua').write_text('changed opponent')
            self.assertEqual((python / 'fa_env/model.py').read_text(), 'original python')
            self.assertEqual((python / 'fa_env/blueprint_vocab.json').read_text(), '{}')
            self.assertEqual((python.parent / 'agentbridge.lua').read_text(), 'original bridge')
            self.assertEqual((python.parent / 'fa_overrides/schook/test.lua').read_text(), 'original hook')
            self.assertEqual((frozen_mod / 'lua/test.lua').read_text(), 'original opponent')
            self.assertFalse((python / '__pycache__').exists())


if __name__ == '__main__':
    unittest.main()
