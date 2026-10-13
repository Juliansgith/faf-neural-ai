"""Raw game recordings: training data that does not depend on any network design.

Every decision is saved with the observation message exactly as the game sent it, the commands sent in
reply and the answers to queries, so any future encoder (a new observation, a new action space, a bigger
network) can be trained offline from games already played. One gzip JSON-lines file per game:

    {"hello": {map, armies, ...}, "meta": {...}}          static game info (blueprints: see below)
    {"step": {"tick", "choice", "sent", "queries", "answers"}, "msg": {...raw observation message...}}
    ...
    {"end": {...episode summary...}}                      missing if the game crashed

Blueprints are identical in every game of a run and large, so they are written once per folder
(blueprints.json.gz).

    for hello, steps, end in read_games(folder): ...
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterator


class Recorder:
    def __init__(self, folder: Path, name: str, extra_meta: dict | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"{name}.jsonl.gz"
        self.file = None
        self.steps = 0
        self.extra_meta = dict(extra_meta or {})

    def start(self, info: dict, meta: dict) -> None:
        blueprints = self.folder / "blueprints.json.gz"
        if not blueprints.exists() and info.get("blueprints"):
            tmp = self.folder / f"blueprints.{os.getpid()}.tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(info["blueprints"], f, separators=(",", ":"))
            os.replace(tmp, blueprints)             # workers may race here; the content is identical
        static = {k: v for k, v in info.items() if k not in ("blueprints", "_bp_index")}
        self.file = gzip.open(self.path, "wb", compresslevel=5)
        self._line({"hello": static, "meta": {**meta, **self.extra_meta}})

    def step(self, raw_message: bytes | None, record: dict) -> None:
        if self.file is None:
            return
        head = json.dumps({"step": record}, separators=(",", ":")).encode("utf-8")[:-1]
        msg = (raw_message or b"null").strip()
        self.file.write(head + b',"msg":' + msg + b"}\n")   # the game's own JSON, not re-encoded
        self.steps += 1

    def end(self, summary: dict | None) -> None:
        if self.file is None:
            return
        self._line({"end": summary or {}})
        self.close()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _line(self, obj: dict) -> None:
        self.file.write(json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n")


def read_game(path: Path) -> tuple[dict, list[dict], dict | None]:
    """(hello record, step records, end summary or None if the game did not finish)."""
    hello, steps, end = {}, [], None
    with gzip.open(path, "rb") as f:
        for line in f:
            rec = json.loads(line)
            if "hello" in rec:
                hello = rec
            elif "step" in rec:
                steps.append(rec)
            elif "end" in rec:
                end = rec["end"]
    return hello, steps, end


def read_blueprints(folder: Path) -> list[dict]:
    with gzip.open(Path(folder) / "blueprints.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def read_games(folder: Path, finished_only: bool = True) -> Iterator[tuple[dict, list[dict], dict | None]]:
    for path in sorted(Path(folder).glob("*.jsonl.gz")):
        try:
            hello, steps, end = read_game(path)
        except (OSError, EOFError, json.JSONDecodeError):
            continue                                # a game cut off mid-write
        if finished_only and end is None:
            continue
        yield hello, steps, end
