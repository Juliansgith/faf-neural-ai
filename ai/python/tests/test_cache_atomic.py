import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fa_train.tools_bc import _write_cache_atomic

class AtomicCacheTest(unittest.TestCase):
    def test_complete_archive_only_appears_after_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'game.v3.npz';native=np.savez_compressed
            def observe(stream,**arrays):
                self.assertFalse(target.exists())
                native(stream,**arrays)
                self.assertFalse(target.exists())
            with patch('fa_train.tools_bc.np.savez_compressed',side_effect=observe):
                _write_cache_atomic(target,{'data':np.arange(5)})
            with np.load(target) as data:np.testing.assert_array_equal(data['data'],np.arange(5))
            self.assertEqual(list(Path(directory).iterdir()),[target])
    def test_failure_preserves_existing_archive_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'game.v3.npz';_write_cache_atomic(target,{'old':np.arange(3)})
            previous=target.read_bytes()
            def fail(stream,**arrays):stream.write(b'partial');raise OSError('disk failure')
            with patch('fa_train.tools_bc.np.savez_compressed',side_effect=fail):
                with self.assertRaises(OSError):_write_cache_atomic(target,{'new':np.arange(4)})
            self.assertEqual(target.read_bytes(),previous)
            self.assertEqual(list(Path(directory).iterdir()),[target])

if __name__=='__main__':unittest.main()
