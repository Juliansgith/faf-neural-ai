"""Training dashboard: a local web page with live, interactive charts of the tools runs.

    python tools/dashboard.py [--runs C:/faf/_runs] [--port 8770] [--open]

Serves http://127.0.0.1:<port>/ (this machine only). It only reads the run folders (night.log, updates.jsonl,
episodes.jsonl, events.jsonl, report.json, checkpoints), so it can run next to a training run. The page
(tools/dashboard.html) refreshes by itself while a run is going. Checkpoint statistics (the "network" tab) are
computed on the CPU and cached per file.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = Path(r"C:\faf\_runs")

STATUS_RE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] ([\d.]+) decisions/s, ([\d.]+) game-s/s, update (\d+), episodes (\d+), "
                       r"restarts (\d+)")
STEP_RE = re.compile(r"^step (\d+): loss ([\d.]+); held-out (.*) \((\d+) s\)")
PAIR_RE = re.compile(r"([a-z_]+) ([\d.]+)")
UPDATE_KEYS = ("update", "time", "stage", "samples", "seconds", "lr", "ent_coef", "entropy", "approx_kl", "clipfrac",
               "pg_loss", "v_loss", "distill_loss", "distill_beta", "belief_loss", "explained_variance", "shaping",
               "epochs_run", "micro_seqs", "peak_gb", "temperature", "planned_updates", "return_std",
               "skipped_minibatches")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def runs_in(root: Path) -> list[dict]:
    out = []
    for d in root.iterdir() if root.exists() else []:
        if not d.is_dir():
            continue
        files = [d / n for n in ("night.log", "updates.jsonl", "episodes.jsonl")]
        present = [f for f in files if f.exists()]
        if not present:
            continue
        mtime = max(f.stat().st_mtime for f in present)
        out.append({"name": d.name, "mtime": mtime, "modified": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                    "tools": (d / "updates.jsonl").exists() or (d / "tools_cache").exists()})
    out.sort(key=lambda r: -r["mtime"])
    return out


def run_data(run: Path) -> dict:
    log = (run / "night.log").read_text(encoding="utf-8", errors="replace").splitlines() if (run / "night.log").exists() else []
    updates = [{k: u.get(k) for k in UPDATE_KEYS} | {k: v for k, v in u.items() if k.startswith("entropy_")}
               for u in read_jsonl(run / "updates.jsonl")]
    episodes = []
    for e in read_jsonl(run / "episodes.jsonl"):
        episodes.append({
            "time": e.get("time"), "update": e.get("update"), "mode": e.get("mode"), "eval": bool(e.get("eval")),
            "tag": e.get("tag"), "ai": e.get("ai"), "faction": e.get("ai_faction"), "seed": e.get("seed"),
            "outcome": e.get("outcome"), "minutes": e.get("minutes"), "return": e.get("return"),
            "comp": e.get("components") or {}, "destroyed": e.get("mass_destroyed"), "lost": e.get("mass_lost"),
            "adv": e.get("value_advantage"), "decisions": e.get("decisions"), "stage": e.get("stage"),
            "income": e.get("mass_income"), "units": e.get("own_units"), "worker": e.get("worker"),
            "wall": e.get("wall_seconds")})
    # status lines carry a clock time only: date them from the first dated record, and step a day on wraparound
    first_dates = [x["time"] for x in episodes[:1] + updates[:1] if x.get("time")]
    base = datetime.fromisoformat(min(first_dates)).date() if first_dates else datetime.fromtimestamp(
        (run / "night.log").stat().st_mtime).date() if (run / "night.log").exists() else datetime.now().date()
    status, clone, header, last_clock, day = [], [], [], None, 0
    for line in log:
        m = STATUS_RE.match(line)
        if m:
            clock = datetime.strptime(m.group(1), "%H:%M:%S").time()
            if last_clock is not None and clock < last_clock and (datetime.combine(base, last_clock) -
                                                                  datetime.combine(base, clock)) > timedelta(hours=6):
                day += 1
            last_clock = clock
            status.append({"t": datetime.combine(base + timedelta(days=day), clock).isoformat(),
                           "decisions_s": float(m.group(2)), "game_s": float(m.group(3)), "update": int(m.group(4)),
                           "episodes": int(m.group(5)), "restarts": int(m.group(6))})
            continue
        m = STEP_RE.match(line)
        if m:
            row = {"step": int(m.group(1)), "loss": float(m.group(2)), "seconds": int(m.group(4))}
            row.update({k: float(v) for k, v in PAIR_RE.findall(m.group(3))})
            clone.append(row)
            continue
        if line.startswith(("tools night run until", "run finished", "size ", "cache:", "report:", "*** promoted",
                            "phase ")) or "failed" in line:
            header.append(line[:300])
    events = [{"time": e.get("time"), "event": e.get("event"), "stage": e.get("stage"), "update": e.get("update"),
               "error": (e.get("error") or "").strip().splitlines()[-1][:200] if e.get("error") else None}
              for e in read_jsonl(run / "events.jsonl")]
    report = {}
    if (run / "report.json").exists():
        try:
            report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        except ValueError:
            report = {}
    clone_from = None
    ref = (report.get("config") or {}).get("clone") or (report.get("config") or {}).get("resume")
    if not clone and ref:                          # cloned in another run: show that run's cloning curve
        other = Path(ref).parent.parent
        if other.is_dir() and other != run and (other / "night.log").exists():
            for line in (other / "night.log").read_text(encoding="utf-8", errors="replace").splitlines():
                m = STEP_RE.match(line)
                if m:
                    row = {"step": int(m.group(1)), "loss": float(m.group(2)), "seconds": int(m.group(4))}
                    row.update({k: float(v) for k, v in PAIR_RE.findall(m.group(3))})
                    clone.append(row)
            clone_from = other.name if clone else None
    stop_at = None
    for line in log:
        m = re.match(r"tools night run until (\d{4}-\d\d-\d\d \d\d:\d\d)", line)
        if m:
            stop_at = m.group(1)
    return {"name": run.name, "status": status, "updates": updates, "episodes": episodes, "clone": clone,
            "events": events, "header": header[-40:], "config": report.get("config") or {}, "stop_at": stop_at,
            "clone_from": clone_from,
            "now": datetime.now().isoformat(timespec="seconds"),
            "checkpoints": sorted(p.name for p in (run / "checkpoints").glob("*.pt")) if (run / "checkpoints").exists() else []}


# ---------------------------------------------------------------------- the network ("neurons")
_weights_lock = threading.Lock()
_state_cache: dict = {}


def _state(path: Path):
    """The model tensors of a checkpoint (CPU), cached per (path, mtime)."""
    import torch
    key = (str(path), path.stat().st_mtime)
    with _weights_lock:
        if key not in _state_cache:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = ck.get("model") if isinstance(ck, dict) else None
            _state_cache[key] = ({k: v.float() for k, v in model.items() if hasattr(v, "float")} if model else {},
                                 {k: ck.get(k) for k in ("update", "episodes", "stage", "size") if isinstance(ck, dict)})
            while len(_state_cache) > 12:
                _state_cache.pop(next(iter(_state_cache)))
        return _state_cache[key]


def _group(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "transformer" and len(parts) > 2:
        return f"transformer.{parts[2]}"
    if parts[0] == "map_blocks" and len(parts) > 1:
        return f"map_blocks.{parts[1]}"
    return parts[0]


def checkpoints_of(run: Path) -> list[tuple[str, Path]]:
    """The run's checkpoints in training order, the clone it started from first (when known)."""
    out = []
    data = json.loads((run / "report.json").read_text(encoding="utf-8")) if (run / "report.json").exists() else {}
    cfg = data.get("config") or {}
    ref = cfg.get("clone") or cfg.get("resume")
    ck_dir = run / "checkpoints"
    if (ck_dir / "tools_bc.pt").exists():
        out.append(("clone", ck_dir / "tools_bc.pt"))
    elif ref and Path(ref).exists() and "tools_bc" in Path(ref).name:
        out.append(("clone", Path(ref)))
    if ck_dir.exists():
        numbered = sorted(ck_dir.glob("ckpt_*.pt"))
        out += [(p.stem.replace("ckpt_", "update "), p) for p in numbered]
        for extra in ("stageB_start.pt", "latest.pt"):
            if (ck_dir / extra).exists():
                out.append((extra.replace(".pt", ""), ck_dir / extra))
    return out


def network_data(run: Path) -> dict:
    cks = checkpoints_of(run)
    if not cks:
        return {"error": "no checkpoints in this run"}
    states = [(label, *_state(path)) for label, path in cks]
    ref_label, ref, _ = states[0]
    layers = []
    for name, w in ref.items():
        if w.dim() == 0:
            continue
        row = {"name": name, "group": _group(name), "shape": list(w.shape), "params": int(w.numel()),
               "norm": {}, "change": {}, "mean_abs": {}}
        for label, st, _ in states:
            v = st.get(name)
            if v is None or v.shape != w.shape:
                continue
            row["norm"][label] = float(v.norm())
            row["mean_abs"][label] = float(v.abs().mean())
            row["change"][label] = float((v - w).norm() / (w.norm() + 1e-8))
        layers.append(row)
    groups: dict = {}
    for l in layers:
        g = groups.setdefault(l["group"], {"group": l["group"], "params": 0})
        g["params"] += l["params"]
    return {"reference": ref_label, "checkpoints": [s[0] for s in states],
            "meta": {s[0]: s[2] for s in states}, "layers": layers,
            "groups": sorted(groups.values(), key=lambda g: -g["params"]),
            "total_params": sum(l["params"] for l in layers)}


def histogram(run: Path, layer: str, bins: int = 60) -> dict:
    cks = checkpoints_of(run)
    if not cks:
        return {"error": "no checkpoints"}
    first, last = _state(cks[0][1])[0], _state(cks[-1][1])[0]
    a, b = first.get(layer), last.get(layer)
    if a is None or b is None:
        return {"error": f"no layer {layer}"}
    import torch
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if not math.isfinite(lo) or hi <= lo:
        hi = lo + 1e-6
    edges = torch.linspace(lo, hi, bins + 1)
    ha = torch.histc(a.flatten(), bins=bins, min=lo, max=hi)
    hb = torch.histc(b.flatten(), bins=bins, min=lo, max=hi)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {"layer": layer, "centers": centers, "first": ha.tolist(), "last": hb.tolist(),
            "first_label": cks[0][0], "last_label": cks[-1][0]}


# ---------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    runs_root: Path = DEFAULT_RUNS

    def log_message(self, *args):                  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _run_path(self, name: str) -> Path | None:
        p = (self.runs_root / unquote(name)).resolve()
        return p if p.parent == self.runs_root.resolve() and p.is_dir() else None

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json(runs_in(self.runs_root))
            elif url.path == "/api/run":
                run = self._run_path(q.get("name", ""))
                self._json(run_data(run) if run else {"error": "unknown run"}, 200 if run else 404)
            elif url.path == "/api/network":
                run = self._run_path(q.get("name", ""))
                self._json(network_data(run) if run else {"error": "unknown run"}, 200 if run else 404)
            elif url.path == "/api/hist":
                run = self._run_path(q.get("name", ""))
                self._json(histogram(run, q.get("layer", "")) if run else {"error": "unknown run"})
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:                   # a half-written file mid-run must not take the page down
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(DEFAULT_RUNS))
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args()
    Handler.runs_root = Path(a.runs)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"training dashboard on {url} (Ctrl+C to stop); runs from {a.runs}", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
