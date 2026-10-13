"""FAEnv: a gym-style environment over the agent bridge. One game process per episode.

    env = FAEnv(GameConfig(map="SCMP_016", faction=1), step_ticks=10)
    obs, info = env.reset()
    while True:
        obs, reward, terminated, truncated, step_info = env.step(actions)
        if terminated or truncated:
            break
    env.close()

Observations and actions are plain dicts (see ai/agentbridge.lua for the exact shapes); featurize.py
turns observations into numpy arrays. There is deliberately no fixed action space: an action is any
dict the Lua script understands, e.g.

    {"type": "move", "units": ["1048577"], "pos": [120.5, 88.0]}
    {"type": "build", "units": [acu_id], "bp": "ueb1103", "pos": [x, z]}
    {"type": "factory_build", "units": [factory_id], "bp": "uel0201", "count": 5}

The result of each action (ok / error) comes back in `step_info["results"]` on the next step.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

from .bridge import BridgeClosed, BridgeServer, GameConfig, GameProcess, minimize_windows, unpin_threads

RewardFn = Callable[[dict | None, dict, dict], float]


class GameError(RuntimeError):
    """The game reported a fatal error (script failed to load, watchdog fired) and is exiting."""


def army(obs: dict, index: int) -> dict | None:
    for entry in obs.get("armies", []):
        if entry.get("index") == index:
            return entry
    return None


def validate_army_ownership(info: dict, config: GameConfig, indices: Iterable[int]) -> None:
    """Fail before play if launch hooks gave an AI control of a neural army."""
    expected = 'AI' if config.player_ai or '/agentallai' in config.extra_args else 'Human'
    for index in indices:
        entry = army(info, index)
        if not entry or entry.get('civilian') or entry.get('brainType') != expected:
            actual = entry.get('brainType') if entry else 'missing'
            raise GameError(f'Army {index} ownership mismatch: expected {expected}, got {actual}; '
                            'refusing a contaminated training/evaluation game')


def default_reward(previous: dict | None, obs: dict, info: dict) -> float:
    """Change in (enemy mass destroyed - own mass lost) / 1000, plus +/-10 when the game is decided.

    A starting point only: sparse win/loss plus a dense trade-efficiency term. Replace it for real
    training (FAEnv(reward_fn=...)).
    """
    me = army(obs, obs.get("me", 1))
    if me is None or "stats" not in me:
        return 0.0

    def trade(o: dict | None) -> float:
        a = army(o, o.get("me", 1)) if o else None
        if not a or "stats" not in a:
            return 0.0
        s = a["stats"]
        return s.get("Enemies_MassValue_Destroyed", 0.0) - s.get("Units_MassValue_Lost", 0.0)

    reward = (trade(obs) - trade(previous)) / 1000.0
    outcome = info.get("outcome")
    if outcome == "victory":
        reward += 10.0
    elif outcome == "defeat":
        reward -= 10.0
    return reward


def outcome_of(obs: dict) -> str | None:
    """'victory', 'defeat' or None (undecided), from the agent army's point of view."""
    me_index = obs.get("me", 1)
    me = army(obs, me_index)
    if me is None:
        return None
    if me.get("defeated"):
        return "defeat"
    enemies = [a for a in obs.get("armies", []) if a.get("relation") == "enemy" and not a.get("civilian")]
    if enemies and all(a.get("defeated") for a in enemies):
        return "victory"
    if me.get("status") == "Victory":
        return "victory"
    return None


class FAEnv:
    def __init__(
        self,
        config: GameConfig | None = None,
        step_ticks: int = 10,
        max_ticks: int = 10 * 60 * 30,
        reward_fn: RewardFn = default_reward,
        omniscient: bool = False,
        armies: Iterable[int] = (1,),
        heightmap_size: int = 64,
        message_timeout: float = 300.0,
        privileged: bool = False,
        capture: bool = False,
        shadow: bool = False,
        views: Iterable[int] = (),
    ):
        self.config = config or GameConfig()
        self.privileged = privileged
        self.capture = capture          # the bridge records another AI's orders to our army (obs["orders"])
        # with capture: the other AI's orders are recorded but not given -- it watches the agent play and says
        # what it would do (labels for the agent's own games). `set_shadow` switches during a game.
        self.shadow = shadow
        self._config_update: dict = {}
        self.step_ticks = int(step_ticks)
        self.max_ticks = int(max_ticks)
        self.reward_fn = reward_fn
        self.omniscient = omniscient
        self.armies = list(armies)
        self.views = list(views)        # more armies observed each step (self-play): obs["views"][str(index)]
        self.heightmap_size = heightmap_size
        self.message_timeout = message_timeout

        self.server: BridgeServer | None = None
        self.game: GameProcess | None = None
        self.info: dict = {}          # the hello message's static info
        self.last_obs: dict | None = None
        self.last_message: dict | None = None
        self.stats = {"steps": 0, "wall": 0.0, "agent_wall": 0.0}
        self._step_started = 0.0

    # ------------------------------------------------------------------ episode control
    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self.close()
        if seed is not None:
            self.config.seed = seed
        self.server = BridgeServer()
        self.game = GameProcess(self.config, self.server.port)
        t0 = time.monotonic()
        self.server.accept(self.config.connect_timeout, alive=self.game.alive)

        hello = self._recv()
        if hello.get("type") == "error":
            raise RuntimeError(f"agent bridge error: {hello.get('errors')}")
        if hello.get("type") != "hello":
            raise RuntimeError(f"expected a hello message, got {hello.get('type')!r}")
        self.info = hello.get("info") or {}
        try:
            validate_army_ownership(self.info, self.config, self.armies)
        except GameError:
            self.close()
            raise
        self.info["load_seconds"] = time.monotonic() - t0
        self.info["log_path"] = str(self.game.log_path)
        self.game.loaded = True
        try:        # a safety net against one-CPU main threads (see unpin_threads)
            self.info["unpinned_threads"] = unpin_threads(self.game.process.pid)
        except Exception as exc:        # never worth failing a game over
            self.info["unpin_error"] = repr(exc)
        if self.config.minimized:       # the sim is running, so the D3D device exists: safe to minimize now
            self.info["minimized"] = minimize_windows(self.game.process.pid)
        if hello.get("errors"):
            self.info["errors"] = hello["errors"]

        config = {"armies": self.armies, "omniscient": self.omniscient, "heightmapSize": self.heightmap_size,
                  "privileged": self.privileged}
        if self.capture:
            config["capture"] = True
            config["shadow"] = bool(self.shadow)
        if self.views:
            config["views"] = self.views
        self._config_update = {}
        self.server.send({"config": config, "wait": 1})
        message = self._recv()
        self.last_message = message
        self.last_obs = message.get("obs") or {}
        self.stats = {"steps": 0, "wall": 0.0, "agent_wall": 0.0, "t0": time.monotonic()}
        self._step_started = time.monotonic()
        return self.last_obs, self.info

    def step(self, actions: Iterable[dict] = (), queries: Iterable[dict] = (), wait: int | None = None):
        if self.server is None:
            raise RuntimeError("call reset() first")
        sent_at = time.monotonic()
        self.stats["agent_wall"] += sent_at - self._step_started

        message = {"actions": list(actions), "queries": list(queries), "wait": int(wait or self.step_ticks)}
        if self._config_update:
            message["config"], self._config_update = self._config_update, {}
        self.server.send(message)
        message = self._recv()
        self._step_started = time.monotonic()
        self.stats["wall"] += self._step_started - sent_at
        self.stats["steps"] += 1

        previous = self.last_obs
        obs = message.get("obs") or {}
        self.last_obs, self.last_message = obs, message

        outcome = outcome_of(obs)
        terminated = bool(message.get("gameOver")) or outcome is not None
        truncated = not terminated and message.get("tick", 0) >= self.max_ticks
        result = message.get("result") or {}
        step_info = {
            "tick": message.get("tick"),
            "outcome": outcome,
            "results": result.get("actions", []),
            "queries": result.get("queries", []),
            "errors": message.get("errors", []),
            "timing": message.get("timing", {}),
        }
        reward = self.reward_fn(previous, obs, step_info) if self.reward_fn else 0.0
        return obs, reward, terminated, truncated, step_info

    def set_shadow(self, on: bool) -> None:
        """From the next step on: the other AI's orders to our army are only recorded (True) or given (False)."""
        self.shadow = bool(on)
        self._config_update["shadow"] = self.shadow

    def query(self, queries: Iterable[dict]) -> list:
        """Run queries (e.g. find_build_site) and advance one tick to get their answers."""
        _, _, _, _, info = self.step(queries=queries, wait=1)
        return info["queries"]

    def close(self) -> None:
        if self.server is not None:
            try:
                self.server.send({"cmd": "quit"})
            except (BridgeClosed, OSError):
                pass
            self.server.close()
            self.server = None
        if self.game is not None:
            self.game.stop()
            self.game = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ helpers
    def _recv(self) -> dict:
        try:
            message = self.server.recv(timeout=self.message_timeout)
            if message.get("type") == "error":
                # The game reports a fatal problem (e.g. the watchdog's Lua traceback) and then exits.
                raise GameError("\n".join(message.get("errors") or ["unknown error"]))
            return message
        except BridgeClosed:
            code = self.game.process.poll() if self.game else None
            raise BridgeClosed(f"the game closed the connection (exit code {code}); log: "
                               f"{self.game.log_path if self.game else '?'}") from None

    def my_units(self, obs: dict | None = None) -> list[dict]:
        obs = obs or self.last_obs or {}
        me = obs.get("me", 1)
        return [u for u in obs.get("units", []) if u.get("army") == me]

    def blueprint(self, bp_id: str) -> dict | None:
        index = self.info.get("_bp_index")
        if index is None:
            index = {bp["id"]: bp for bp in self.info.get("blueprints", [])}
            self.info["_bp_index"] = index
        return index.get(bp_id)
