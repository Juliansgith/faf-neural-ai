"""MacroEnv: a gymnasium environment with featurized observations and a small macro-action space.

The usual first step for RTS reinforcement learning: the policy chooses *what* to do every couple of
seconds, scripted micro decides *how*. Observations come from Featurizer; `action_mask` in the
observation marks the macros that can act right now, for masked policies.

    env = MacroEnv(GameConfig(map="SCMP_016", ai_faction=2), decision_ticks=20)
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample(mask=obs["action_mask"]))

Extend MACROS (and _plan / _mask) to give the policy more to decide; swap in your own reward with
reward_fn (see env.default_reward).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agents import classify, distance, resolve_blueprints
from .bridge import GameConfig
from .env import FAEnv
from .featurize import SPATIAL_CHANNELS, UNIT_FEATURES, Featurizer

MACROS = (
    "noop", "build_mex", "build_power", "build_factory",
    "train_engineer", "train_tank", "attack", "retreat", "assist",
)
N_SCALARS = 11


class MacroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: GameConfig | None = None, decision_ticks: int = 20, max_minutes: float = 20.0,
                 map_size: int = 64, max_units: int = 256, **fa_kwargs):
        self.fa = FAEnv(config or GameConfig(), step_ticks=decision_ticks, max_ticks=int(max_minutes * 600),
                        **fa_kwargs)
        self.decision_ticks = decision_ticks
        self.map_size, self.max_units = map_size, max_units
        self.action_space = spaces.Discrete(len(MACROS))
        self.observation_space = spaces.Dict({
            "map": spaces.Box(-np.inf, np.inf, (len(SPATIAL_CHANNELS), map_size, map_size), np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, (N_SCALARS,), np.float32),
            "units": spaces.Box(-np.inf, np.inf, (max_units, len(UNIT_FEATURES)), np.float32),
            "unit_bp": spaces.Box(0, 1 << 16, (max_units,), np.int64),
            "unit_mask": spaces.MultiBinary(max_units),
            "action_mask": spaces.MultiBinary(len(MACROS)),
        })

    # ------------------------------------------------------------------ gymnasium API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self.fa.reset(seed=seed)
        self.info = info
        self.bps = {bp["id"]: bp for bp in info["blueprints"]}
        self.featurizer = Featurizer(info, self.map_size, self.max_units)
        self.me = obs["me"]
        armies = {a["index"]: a for a in info["armies"]}
        self.start = tuple(armies[self.me]["start"])
        self.targets = [tuple(a["start"]) for a in info["armies"]
                        if a["relation"] == "enemy" and not a["civilian"]] or [self.start]
        self.deposits = [(d["x"], d["z"]) for d in info["map"]["deposits"] if d["type"] == "mass"]
        self.chosen = None
        self.claimed: dict[tuple, int] = {}
        self.obs = obs
        return self._encode(obs), {"tick": obs.get("tick", 0), "vocab_size": self.featurizer.vocab_size}

    def step(self, action):
        macro = MACROS[int(action)]
        actions, site_request = self._plan(macro, self.obs)
        reward_total, wait = 0.0, self.decision_ticks
        if site_request is not None:
            # place the structure: one tick for the site query, then the build order
            unit_id, bp_id = site_request
            query = {"type": "find_build_site", "bp": bp_id, "pos": list(self.start), "radius": 40}
            obs, reward, terminated, truncated, info = self.fa.step(queries=[query], wait=1)
            reward_total += reward
            wait -= 1
            if terminated or truncated:
                self.obs = obs
                return self._encode(obs), reward_total, terminated, truncated, self._info(info, macro)
            answer = (info["queries"] or [{}])[0]
            if answer.get("ok") and answer.get("value"):
                actions.append({"type": "build", "units": [unit_id], "bp": bp_id, "pos": answer["value"]})
        obs, reward, terminated, truncated, info = self.fa.step(actions, wait=max(1, wait))
        self.obs = obs
        return self._encode(obs), reward_total + reward, terminated, truncated, self._info(info, macro)

    def close(self):
        self.fa.close()

    # ------------------------------------------------------------------ macros
    def _roles(self, obs):
        own = [u for u in obs.get("units", []) if u.get("army") == self.me and not u.get("radar")]
        roles = classify(own, self.bps)
        if roles["acu"] and self.chosen is None:
            self.chosen = resolve_blueprints(self.bps, roles["acu"][0]["bp"])
        return roles

    def _idle_builder(self, roles, near=None):
        builders = [u for u in roles["engineer"] if u.get("idle") and u.get("frac", 1) >= 1]
        builders += [u for u in roles["acu"] if u.get("idle")]
        if not builders:
            return None
        if near is None:
            return builders[0]
        return min(builders, key=lambda u: distance((u["x"], u["z"]), near))

    def _free_deposit(self, roles, tick):
        built = {(round(m["x"]), round(m["z"])) for m in roles["mex"]}
        free = [d for d in self.deposits
                if (round(d[0]), round(d[1])) not in built and tick - self.claimed.get(d, -10 ** 9) > 600]
        return min(free, key=lambda d: distance(d, self.start)) if free else None

    def _mask(self, obs) -> np.ndarray:
        roles = self._roles(obs)
        chosen = self.chosen or {}
        tick = obs.get("tick", 0)
        builder = self._idle_builder(roles) is not None
        idle_factory = any(f.get("idle") and f.get("frac", 1) >= 1 for f in roles["factory"])
        army = any(t.get("frac", 1) >= 1 for t in roles["tank"])
        mask = np.zeros(len(MACROS), np.int8)
        mask[0] = 1
        mask[1] = builder and chosen.get("mex") is not None and self._free_deposit(roles, tick) is not None
        mask[2] = builder and chosen.get("pgen") is not None
        mask[3] = builder and chosen.get("factory") is not None
        mask[4] = idle_factory and chosen.get("engineer") is not None
        mask[5] = idle_factory and chosen.get("tank") is not None
        mask[6] = army
        mask[7] = army
        mask[8] = bool(roles["factory"]) and any(u.get("idle") for u in roles["engineer"])
        return mask

    def _plan(self, macro, obs):
        """Orders for one macro; also returns (unit, blueprint) when a build site must be found first."""
        roles = self._roles(obs)
        chosen = self.chosen or {}
        tick = obs.get("tick", 0)
        actions = []
        if macro == "build_mex":
            deposit = self._free_deposit(roles, tick)
            builder = self._idle_builder(roles, deposit)
            if deposit and builder and chosen.get("mex"):
                self.claimed[deposit] = tick
                actions.append({"type": "build", "units": [builder["id"]], "bp": chosen["mex"], "pos": list(deposit)})
        elif macro in ("build_power", "build_factory"):
            bp = chosen.get("pgen" if macro == "build_power" else "factory")
            builder = self._idle_builder(roles, self.start)
            if bp and builder:
                return actions, (builder["id"], bp)
        elif macro in ("train_engineer", "train_tank"):
            bp = chosen.get("engineer" if macro == "train_engineer" else "tank")
            factories = [f for f in roles["factory"] if f.get("idle") and f.get("frac", 1) >= 1]
            if bp and factories:
                actions.append({"type": "factory_build", "units": [factories[0]["id"]], "bp": bp,
                                "count": 1 if macro == "train_engineer" else 2})
        elif macro in ("attack", "retreat"):
            army = [t["id"] for t in roles["tank"] if t.get("frac", 1) >= 1]
            if army:
                if macro == "retreat":
                    actions.append({"type": "move", "units": army, "pos": list(self.start)})
                else:
                    enemies = [u for u in obs.get("units", []) if u.get("army") != self.me
                               and self._relation(obs, u.get("army")) == "enemy"]
                    if enemies:
                        target = min(enemies, key=lambda u: distance((u["x"], u["z"]), self.start))
                        pos = [target["x"], target["z"]]
                    else:
                        pos = list(self.targets[0])
                    actions.append({"type": "attack_move", "units": army, "pos": pos})
        elif macro == "assist":
            engineers = [u["id"] for u in roles["engineer"] if u.get("idle")]
            if engineers and roles["factory"]:
                actions.append({"type": "guard", "units": engineers, "target": roles["factory"][0]["id"]})
        return actions, None

    @staticmethod
    def _relation(obs, army_index):
        for a in obs.get("armies", []):
            if a["index"] == army_index:
                return a["relation"]
        return "enemy"

    # ------------------------------------------------------------------ encoding
    def _encode(self, obs):
        spatial, scalars, (units, bp_ids, mask) = self.featurizer(obs)
        return {
            "map": spatial, "scalars": scalars, "units": units, "unit_bp": bp_ids,
            "unit_mask": mask.astype(np.int8), "action_mask": self._mask(obs),
        }

    @staticmethod
    def _info(step_info, macro):
        return {"tick": step_info.get("tick"), "macro": macro, "outcome": step_info.get("outcome"),
                "errors": step_info.get("errors"), "results": step_info.get("results")}
