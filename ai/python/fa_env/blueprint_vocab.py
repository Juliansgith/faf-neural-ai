"""Stable embedding IDs shared by live games and modded teacher recordings."""
import hashlib
import json
from pathlib import Path

VOCAB_PATH = Path(__file__).with_suffix('.json')
_DATA = json.loads(VOCAB_PATH.read_text(encoding='utf-8'))
if len(_DATA['ids']) != len(set(_DATA['ids'])) or len(_DATA['ids']) >= 1024:
    raise ValueError('Blueprint vocabulary must be unique and fit the model embedding')
BLUEPRINT_INDEX = {name: index + 1 for index, name in enumerate(_DATA['ids'])}
VOCAB_SHA256 = hashlib.sha256(VOCAB_PATH.read_bytes()).hexdigest()


def blueprint_indices(blueprints):
    return {name: BLUEPRINT_INDEX.get(name, 0) for name in blueprints}
