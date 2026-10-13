"""Recordings from FAF replays: human games as training data in the format of our own recorded games.

    python -m fa_train.replay_rec --replays DIR --out RAW [--workers 2] [--keep-work]

Our engine does not replay FAF's games in sync (it diverges within seconds), FAF's own exe does. So each replay is
played by FAF's ForgedAlliance.exe (under Wine on Linux, or wherever bridge.WINE says), with two hooks
(ai/replay_rec/schook, mounted at /schook): the sim loads the agent's observation script (agentbridge.lua) once per
human army and observes the game from that army's side every 20 ticks, as in a training game; the UI plays the
replay at full speed. Nothing gives an order, so the replay stays in sync (the sim still checks the players'
checksums). The observations reach this process through the preferences file (the sim's sync table -> the UI ->
SetPreference/SavePreferences; the sim Lua cannot write files and the game log keeps every byte in memory).

Each human army becomes one recording RAW/<game id>_a<army>.jsonl.gz (fa_env.record.Recorder): hello = the army's
AgentBridge_Hello, steps = its observations with the player's own commands since the previous decision in `sent`
(fa_train.replay_orders: the replay's command stream as raw orders), end = the outcome. Without a label source the
encoder labels a step with ToolKit.commands_from_actions(sent), as for any recorded game; encode_parallel encodes
RAW as usual. Maps missing locally are fetched from FAF's map vault; generated maps (neroxis) are skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fa_env import bridge                                      # noqa: E402
from fa_env.record import Recorder                            # noqa: E402
from fa_train.replay_orders import replay_orders              # noqa: E402

HOOKS = ROOT.parent / "replay_rec" / "schook"
DECISION_TICKS = 20
MAP_URL = "https://content.faforever.com/maps/{}.zip"
UA = {"User-Agent": "faf-neural-ai replay research"}
SLOT = re.compile(r"replayrec_\d+\s*=\s*'((?:[^'\\]|\\.)*)'", re.S)
ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "\n": "\n"}


def unlua(s: str) -> str:
    return re.sub(r"\\(\d{1,3}|.)", lambda m: chr(int(m.group(1))) if m.group(1).isdigit()
                  else ESC.get(m.group(1), m.group(1)), s, flags=re.S)


def read_replay(path: Path) -> tuple[dict, bytes]:
    """(the .fafreplay JSON line, the SCFA replay body)."""
    raw = path.read_bytes()
    head, _, body = raw.partition(b"\n")
    meta = json.loads(head)
    if meta.get("compression") == "zstd":
        import zstandard
        body = zstandard.ZstdDecompressor().decompressobj().decompress(body)
    else:
        import base64
        import zlib
        body = zlib.decompress(base64.b64decode(body)[4:])
    return meta, body


def vault_maps_dir() -> Path:
    """The game's user maps folder (PREFS_DIR is <user>/AppData/Local/Gas Powered Games/<game>, in Wine's too)."""
    return bridge.PREFS_DIR.parents[3] / "Documents" / "My Games" / "Gas Powered Games" / \
        "Supreme Commander Forged Alliance" / "Maps"


def ensure_map(map_file: str, stock: Path) -> bool:
    """The replay's map is here (stock, or fetched from FAF's vault into the user's maps folder)."""
    folder = map_file.strip("/").split("/")[1] if map_file.count("/") >= 2 else ""
    if not folder or folder.lower().startswith("neroxis_map_generator"):
        return False
    if any((stock / f).exists() for f in (folder, folder.upper(), folder.lower())):
        return True
    dest = vault_maps_dir()
    if (dest / folder).exists():
        return True
    from urllib.parse import quote
    data, error = None, None
    # the vault's zip is named after the map's folder, usually lower case; older maps keep their spaces
    for name in dict.fromkeys((folder, folder.lower(), folder.lower().replace(" ", "_"))):
        try:
            with urllib.request.urlopen(urllib.request.Request(MAP_URL.format(quote(name)), headers=UA),
                                        timeout=60) as r:
                data = r.read()
            break
        except Exception as exc:                              # noqa: BLE001 -- try the next spelling
            error = exc
    if data is None:
        print(f"map {folder}: not in the vault ({type(error).__name__}: {error})", flush=True)
        return False
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    return any(p.name.lower() == folder.lower() for p in dest.iterdir())


def play(scfa: Path, work: Path, timeout: float = 1800) -> tuple[list[dict], str | None]:
    """Plays a replay in FAF's exe with the recorder hooks; returns (records ordered by seq, a desync line)."""
    rid = uuid.uuid4().hex[:8]
    log = work / f"{scfa.stem}_{rid}.log"
    init = work / f"init_{rid}.lua"
    b = bridge.game_posix(bridge.DEFAULT_BIN_DIR)
    init.write_text(
        f"InitFileDir = [[{b}]]\n"
        f"dofile([[{b}/init.lua]])\n"
        "local SetProcessPriority = rawget(_G, 'SetProcessPriority')\n"
        "if SetProcessPriority then SetProcessPriority(0x4000) end\n"
        f"table.insert(path, 1, {{ dir = [[{bridge.game_posix(HOOKS)}]], mountpoint = '/schook' }})\n"
        f"table.insert(path, 1, {{ dir = [[{bridge.game_posix(HOOKS.parents[1])}]], mountpoint = '/agentai' }})\n")
    prefs, prefs_path = bridge.make_private_prefs(rid, "replay")
    args = [bridge.game_path(bridge.DEFAULT_BIN_DIR / "ForgedAlliance.exe"), "/init", bridge.game_path(init),
            "/nobugreport", "/nosound", "/nomovie", "/log", bridge.game_path(log), "/windowed", "1024", "768",
            "/prefs", prefs, "/replay", bridge.game_path(scfa)]
    if bridge.WINE:
        env = dict(os.environ, WINEDEBUG="-all", DISPLAY=os.environ.get("FA_ENV_DISPLAY", ":99"),
                   WINEDLLOVERRIDES=bridge.WINE_DLL_OVERRIDES.replace("d3dx9_31,", "d3dx9_31,d3dx9_35,"))
        p = subprocess.Popen(["wine"] + args, cwd=str(bridge.DEFAULT_BIN_DIR), env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, preexec_fn=lambda: os.nice(10))
    else:
        raise RuntimeError("FAF's exe needs a desktop it cannot steal focus from: run this under Wine")
    records, seen = [], set()
    t0 = last_new = last_check = time.monotonic()
    last_mtime, bad = 0, None
    try:
        while time.monotonic() - t0 < timeout and p.poll() is None:
            time.sleep(0.25)
            try:
                mtime = prefs_path.stat().st_mtime_ns
            except OSError:
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    text = prefs_path.read_text(errors="replace")
                except OSError:
                    continue
                for m in SLOT.finditer(text):
                    try:
                        kind, army, tick, seq, payload = unlua(m.group(1)).split("|", 4)
                        if seq in seen:
                            continue
                        data = json.loads(payload)
                    except ValueError:
                        continue                              # read mid-write: the next read has it
                    seen.add(seq)
                    records.append({"kind": kind, "army": int(army), "tick": int(tick), "seq": int(seq),
                                    "data": data})
                    last_new = time.monotonic()
            if records and time.monotonic() - last_new > 30:  # the replay is over
                break
            if time.monotonic() - last_check > 10:
                last_check = time.monotonic()
                text = log.read_text(errors="replace") if log.exists() else ""
                m = re.search(r"Checksum for beat (\d+) mismatched[^\n]*", text)
                if m:
                    bad = m.group(0)
                    break
    finally:
        subprocess.call(["pkill", "-f", prefs])                # the game itself (Wine runs it in its own process)
        p.kill()
        p.wait()
        prefs_path.unlink(missing_ok=True)
        init.unlink(missing_ok=True)
    records.sort(key=lambda r: r["seq"])
    if bad is None and records:
        # the preferences file is a ring of 64 records: a reader that fell behind lost some, and a lost observation
        # would move its orders onto the step before it
        lost = records[-1]["seq"] - len(records)
        if lost:
            bad = f"{lost} of {records[-1]['seq']} records lost (the reader fell behind the ring)"
    return records, bad


def outcome_of(obs: dict, me: int) -> str | None:
    """From the army's status (at the end every army counts as defeated, the winner's status says Victory)."""
    armies = [a for a in obs.get("armies") or [] if not a.get("civilian")]
    mine = next((a for a in armies if a.get("index") == me), {})
    status = {"Victory": "victory", "Defeat": "defeat", "Draw": "draw"}.get(mine.get("status"))
    if status:
        return status
    others = [a for a in armies if a.get("index") != me and a.get("relation") == "enemy"]
    if mine.get("defeated"):
        return "defeat"
    if others and all(a.get("defeated") for a in others):
        return "victory"
    return None


RECLAIM_COMMAND = 19                                          # EUnitCommandType UNITCOMMAND_Reclaim


def orders_per_step(obs: list[dict], orders: list[tuple[int, dict]]) -> list[list[dict]]:
    """Each observation's orders: the player's orders from its tick to the next observation's. A reclaim order
    names only the prop; where it is comes from the next observation, whose engineers carry it as their current
    order's destination (without it the encoder would put the reclaim where the engineers stand)."""
    out, k = [], 0
    for i, o in enumerate(obs):
        t_next = obs[i + 1]["tick"] if i + 1 < len(obs) else 1 << 30
        sent = []
        while k < len(orders) and orders[k][0] < t_next:
            if orders[k][0] >= o["tick"]:
                sent.append(dict(orders[k][1]))
            k += 1
        nxt = obs[i + 1]["data"].get("units") or [] if i + 1 < len(obs) else []
        for order in sent:
            if order["type"] == "reclaim" and "pos" not in order and "target_pos" not in order:
                ids = set(order["units"])
                dest = next((u["dest"] for u in nxt if u.get("id") in ids and u.get("order") == RECLAIM_COMMAND
                             and u.get("dest")), None)
                if dest:
                    order["target_pos"] = [float(dest[0]), float(dest[1])]
        out.append(sent)
    return out


def assemble(gid: str, records: list[dict], orders: dict, out: Path, extra: dict) -> list[Path]:
    """One recording per observed army."""
    written = []
    for army in sorted({r["army"] for r in records if r["kind"] == "H"}):
        hello = next(r["data"] for r in records if r["kind"] == "H" and r["army"] == army)
        obs = [r for r in records if r["kind"] == "O" and r["army"] == army]
        if len(obs) < 10:
            continue
        mine = next((a for a in hello.get("armies") or [] if a.get("index") == army), {})
        enemy = next((a for a in hello.get("armies") or [] if a.get("relation") == "enemy"), {})
        meta = {"map": (hello.get("map") or {}).get("name") or extra.get("map"), "faction": mine.get("faction"),
                "ai": "human", "ai_faction": enemy.get("faction"), "seed": 0, "victory": "demoralization",
                "decision_ticks": DECISION_TICKS, "time_limit_ticks": obs[-1]["tick"] + DECISION_TICKS, "me": army,
                "source": "faf_replay", **extra}
        rec = Recorder(out, f"{gid}_a{army}")
        rec.start(hello, meta)
        for o, sent in zip(obs, orders_per_step(obs, orders.get(army, []))):
            msg = json.dumps({"type": "obs", "tick": o["tick"], "gameOver": False, "obs": o["data"]},
                             separators=(",", ":")).encode()
            rec.step(msg, {"tick": o["tick"], "commands": [], "sent": sent, "queries": [], "answers": [],
                           "results": [], "labels": None})
        result = outcome_of(obs[-1]["data"], army)
        rec.end({"outcome": result, "minutes": round(obs[-1]["tick"] / 600, 1), "source": "faf_replay"})
        written.append(rec.path)
    return written


def process(path: Path, out: Path, work: Path, stock: Path, keep_work: bool = False) -> dict:
    gid = path.name.split(".")[0]
    meta, body = read_replay(path)
    from fafreplay import Parser
    header = Parser().parse_header(body)
    map_file = header.get("map_file")
    map_file = map_file.decode(errors="replace") if isinstance(map_file, bytes) else str(map_file)
    if not ensure_map(map_file, stock):
        return {"id": gid, "ok": False, "why": f"map {map_file}"}
    scfa = work / f"{gid}.SCFAReplay"
    scfa.write_bytes(body)
    t0 = time.monotonic()
    try:
        records, bad = play(scfa, work)
    finally:
        if not keep_work:
            scfa.unlink(missing_ok=True)
    if bad:
        return {"id": gid, "ok": False, "why": bad if "lost" in bad else f"desync: {bad}"}
    orders = replay_orders(body)
    ratings = {}
    for a in (header.get("armies") or {}).values():
        if isinstance(a, dict) and a.get("Human"):
            name = a.get("PlayerName")
            ratings[name.decode(errors="replace") if isinstance(name, bytes) else str(name)] = \
                [round(float(a.get("MEAN") or 0)), round(float(a.get("DEV") or 0))]
    files = assemble(gid, records, orders, out, {"replay_id": int(gid) if gid.isdigit() else gid,
                                                 "ratings": ratings, "map_file": map_file})
    return {"id": gid, "ok": bool(files), "records": len(records), "files": [f.name for f in files],
            "orders": {str(k): len(v) for k, v in orders.items()}, "seconds": round(time.monotonic() - t0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", type=Path, required=True, help="folder of .fafreplay files")
    ap.add_argument("--out", type=Path, required=True, help="recordings folder (encode_parallel --raw)")
    ap.add_argument("--work", type=Path, default=Path.home() / "replay_rec")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stock-maps", type=Path, default=Path(os.path.expanduser("~/fa/FA/maps")))
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.work.mkdir(parents=True, exist_ok=True)
    index = a.out / "replays.jsonl"
    done = set()
    if index.exists():
        done = {json.loads(line)["id"] for line in index.read_text().splitlines() if line.strip()}
    todo = [p for p in sorted(a.replays.glob("*.fafreplay")) if p.name.split(".")[0] not in done]
    print(f"{len(todo)} replays to record ({len(done)} done before)", flush=True)

    def one(p):
        try:
            return process(p, a.out, a.work, a.stock_maps, a.keep_work)
        except Exception as exc:                              # noqa: BLE001 -- one bad replay
            return {"id": p.name.split(".")[0], "ok": False, "why": f"{type(exc).__name__}: {exc}"[:300]}

    with ThreadPoolExecutor(a.workers) as pool, index.open("a") as log:
        for res in pool.map(one, todo):
            log.write(json.dumps(res) + "\n")
            log.flush()
            print(json.dumps(res)[:300], flush=True)


if __name__ == "__main__":
    main()
