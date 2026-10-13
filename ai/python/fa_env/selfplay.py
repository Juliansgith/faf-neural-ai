"""Self-play: two of our networks in one game.

Army 1 and army 2 are both M28 brains in shadow mode: M28 thinks for each, its orders are recorded (the labels)
and swallowed, and each army is played by a ToolsEnv -- a side -- whose bridge is a SideLink. The two sides step
in lockstep (both make both bridge trips every decision): their trips meet here and go to the game as one, every
action and query naming its army, and come back split -- each side its own view of the game (the bridge's
obs.views), its own shadow orders, its own action results and query answers.

    game = SelfPlayGame(GameConfig(map="SCMP_016"), time_limit_minutes=40)
    (obs_a, info_a), (obs_b, info_b) = game.reset()
    (obs_a, r_a, done, info_a), (obs_b, r_b, _, info_b) = game.step(commands_a, commands_b)

Only 1v1 maps: side 2's hello is side 1's with the two armies' relations swapped.
"""
from __future__ import annotations

import copy
import json
import threading

from .bridge import GameConfig
from .env import FAEnv, outcome_of
from .tools_env import ToolsEnv

ARMIES = (1, 2)


class _Line:
    """What a side's recorder takes as the game's raw line (its own view, re-encoded)."""
    last_line: bytes | None = None


class SideLink:
    """A side's bridge: its trips go through the SelfPlayGame."""
    capture = True
    shadow = True           # M28's orders to the side's army are labels, never given (ToolsEnv._tools_result)

    def __init__(self, game: "SelfPlayGame", side: int):
        self.selfplay, self.side = game, side
        self.server = _Line()
        self.config = game.config

    @property
    def game(self):                 # the game process (the worker reads its pid)
        return self.selfplay.fa.game

    def step(self, actions=(), queries=(), wait=None):
        return self.selfplay._trip(self.side, list(actions), list(queries), int(wait or 1))

    def reset(self):
        raise RuntimeError("a self-play side is reset by its SelfPlayGame")

    def set_shadow(self, on: bool) -> None:
        pass

    def close(self) -> None:
        self.selfplay.close()


def side_info(info: dict, army: int) -> dict:
    """The hello as army `army` sees it (1v1: the two armies' relations swapped)."""
    out = copy.deepcopy(info)
    for a in out.get("armies", []):
        if a.get("index") in ARMIES:
            a["relation"] = "self" if a["index"] == army else "enemy"
    return out


class SelfPlayGame:
    def __init__(self, config: GameConfig, decision_ticks: int = 20, time_limit_minutes: float = 20.0,
                 **env_kwargs):
        config.player_ai = "m28ai"           # both armies M28 brains, shadowed: labels for both, neither plays
        config.ai, config.cheat = "m28ai", None
        self.config = config
        self.fa = FAEnv(config, step_ticks=decision_ticks, max_ticks=int(time_limit_minutes * 600), privileged=True,
                        capture=True, shadow=True, armies=ARMIES, views=ARMIES[1:])
        self.sides: list[ToolsEnv] = []
        for side in range(2):
            env = ToolsEnv(config, decision_ticks=decision_ticks, time_limit_minutes=time_limit_minutes,
                           capture=True, shadow=True, **env_kwargs)
            env.fa = SideLink(self, side)
            env.lockstep = True
            self.sides.append(env)
        self._in: list = [None, None]
        self._out: list = [None, None]
        self._barrier = threading.Barrier(2, action=self._merged_trip)
        self._closed = False

    # ------------------------------------------------------------------ the shared bridge
    def _split(self, message: dict) -> tuple[dict, dict, dict, dict]:
        """(obs of army 1, obs of army 2, raw message of 1, of 2) from one bridge message (its obs edited)."""
        obs = message.get("obs") or {}
        views = obs.pop("views", None) or {}
        orders = obs.get("orders") or []
        obs_b = dict(views.get(str(ARMIES[1])) or {})
        obs["orders"] = [o for o in orders if o.get("army", ARMIES[0]) == ARMIES[0]]
        obs_b["orders"] = [o for o in orders if o.get("army") == ARMIES[1]]
        if "privileged" in obs and "privileged" not in obs_b:       # every army's true totals: the same for both
            obs_b["privileged"] = obs["privileged"]
        msg_b = {k: v for k, v in message.items() if k not in ("obs", "result")}
        msg_b["obs"] = obs_b
        return obs, obs_b, message, msg_b

    def _set_lines(self, msg_a: dict, msg_b: dict) -> None:
        self.sides[0].fa.server.last_line = json.dumps(msg_a).encode("utf-8")
        self.sides[1].fa.server.last_line = json.dumps(msg_b).encode("utf-8")

    def _trip(self, side: int, actions: list, queries: list, wait: int):
        self._in[side] = (actions, queries, wait)
        self._barrier.wait()
        out = self._out[side]
        if isinstance(out, BaseException):
            raise out
        return out

    def _merged_trip(self) -> None:
        """Runs once both sides have arrived (in one of their threads)."""
        try:
            (aa, qa, wa), (ab, qb, wb) = self._in
            tag = lambda xs, army: [dict(x, army=army) for x in xs]    # noqa: E731
            _, _, terminated, truncated, info = self.fa.step(tag(aa, ARMIES[0]) + tag(ab, ARMIES[1]),
                                                             tag(qa, ARMIES[0]) + tag(qb, ARMIES[1]),
                                                             wait=min(wa, wb))
            obs_a, obs_b, msg_a, msg_b = self._split(self.fa.last_message)
            self._set_lines(msg_a, msg_b)
            results, answers = info.get("results") or [], info.get("queries") or []
            for side, obs, na, nq, ra, rq in ((0, obs_a, len(aa), len(qa), 0, 0),
                                              (1, obs_b, len(ab), len(qb), len(aa), len(qa))):
                outcome = outcome_of(obs)
                done = bool(terminated) or outcome is not None
                step_info = dict(info, outcome=outcome, results=results[ra:ra + na], queries=answers[rq:rq + nq])
                self._out[side] = (obs, 0.0, done, bool(truncated) and not done, step_info)
        except BaseException as exc:            # both sides raise it
            self._out = [exc, exc]

    # ------------------------------------------------------------------ episode
    def reset(self):
        """Launches the game. Returns ((obs, info) of side 1, (obs, info) of side 2)."""
        _, info = self.fa.reset()
        obs_a, obs_b, msg_a, msg_b = self._split(self.fa.last_message)
        self._set_lines(msg_a, msg_b)
        out_a = self.sides[0].attach(obs_a, info)
        out_b = self.sides[1].attach(obs_b, side_info(info, ARMIES[1]))
        return out_a, out_b

    def step(self, commands_a: list, commands_b: list):
        """One decision of both sides: (side 1's step_tools result, side 2's)."""
        out: list = [None, None]
        err: list = [None]

        def run_b():
            try:
                out[1] = self.sides[1].step_tools(commands_b)
            except BaseException as exc:        # noqa: BLE001 -- handed to the caller's thread
                err[0] = exc
                self._barrier.abort()

        t = threading.Thread(target=run_b, daemon=True)
        t.start()
        try:
            out[0] = self.sides[0].step_tools(commands_a)
        except BaseException:
            self._barrier.abort()
            t.join(timeout=30)
            raise
        t.join()
        if err[0] is not None:
            raise err[0]
        return out[0], out[1]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._barrier.abort()
        except Exception:                       # noqa: BLE001
            pass
        self.fa.close()
