"""Targeted equivalence test of the native observation unit list (AgentBridge_ObserveUnits) against the bridge
script's Lua build, for the unit states a teacher game rarely produces: a loaded transport, tactical and nuke
silos with missiles, a paused factory, abilities switched on, an enhanced commander, enemy units in sight, and
shields (structure, mobile, commanders' personal and bubble ones) at full strength, damaged, knocked down and
recharging, switched off and on again.

    python obs_equivalence_probe.py --exe C:/faf/_enginetest/bin_c/main.exe [--steps 80]

Spawns those units for army 1 with the bridge's eval query (development aid), then steps the game with both unit
lists in every observation (obsCheck) and diffs them. Exit code 0 when every observation matched.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai" / "python"))
sys.path.insert(0, str(HERE))
from fa_env import FAEnv, GameConfig  # noqa: E402
from perf_game import diff_units, same  # noqa: E402

SPAWN = r"""
local brain = ArmyBrains[1]
local x, z = brain:GetArmyStartPos()
local function mk(bp, army, dx, dz)
    return CreateUnitHPR(bp, army, x + dx, GetSurfaceHeight(x + dx, z + dz), z + dz, 0, 0, 0)
end
local made = {}
local transport = mk('uea0107', 1, 12, 12)
local a, b = mk('uel0105', 1, 16, 12), mk('uel0105', 1, 16, 15)
IssueTransportLoad({ a, b }, transport)
local tml = mk('ueb2108', 1, -18, 12)
tml:GiveTacticalSiloAmmo(2)
local nuke = mk('ueb2305', 1, -20, -20)
nuke:GiveNukeSiloAmmo(1)
local factory = mk('ueb0101', 1, 24, -12)
factory:SetPaused(true)
local radar = mk('ueb3101', 1, 28, 2)
-- shields: a UEF T2 generator (bubble), UEF / Aeon T2 and Seraphim T3 mobile shields, commanders with their
-- personal shield (UEF, Aeon) or bubble (UEF field), a Seraphim support commander's shield
-- (T3 power first: shields drain a fresh army's energy and would stay down)
for i = 0, 3 do mk('ueb1301', 1, -34, -30 + 12 * i) end
local shield = mk('ueb4202', 1, -8, -24)
mk('uel0307', 1, -12, -18)
mk('ual0307', 1, -14, -20)
mk('xsl0307', 1, -16, -22)
local fighters = { mk('uea0102', 1, 5, 5), mk('uea0102', 1, 6, 5) }
for _, u in fighters do u:SetFireState(1) end
local acu = brain:GetListOfUnits(categories.COMMAND, false)[1]
if acu then
    pcall(acu.CreateEnhancement, acu, 'AdvancedEngineering')
    pcall(acu.CreateEnhancement, acu, 'Shield')
end
local aeonAcu = mk('ual0001', 1, 8, -12)
pcall(aeonAcu.CreateEnhancement, aeonAcu, 'Shield')
local uefField = mk('uel0301', 1, 10, -14)
pcall(uefField.CreateEnhancement, uefField, 'Shield')
pcall(uefField.CreateEnhancement, uefField, 'ShieldGeneratorField')
local seraSacu = mk('xsl0301', 1, 12, -16)
pcall(seraSacu.CreateEnhancement, seraSacu, 'Shield')
-- enemies in sight of army 1
for i = 1, 4 do mk('ual0201', 2, 30 + 2 * i, -4) end
mk('uab2101', 2, 34, 4)
return 'spawned'
"""

# Shield events: damage every shield by a share of its maximum (1.5: more than it holds -- it goes down and
# recharges), or switch shields off / on (script bit 0 clear / set: FAF's DisableShield / EnableShield). `which`
# picks every n-th shielded unit.
SHIELD_EVENT = r"""
local share, switch, every = %s, %s, %d
local n, i = 0, 0
for _, u in ArmyBrains[1]:GetListOfUnits(categories.ALLUNITS, false) do
    local s = u.MyShield
    if s and not u.Dead then
        i = i + 1
        if math.mod(i, every) == 0 then
            if share > 0 then
                pcall(s.ApplyDamage, s, nil, share * s:GetMaxHealth(), Vector(0, 0, 0), 'Normal', false)
            end
            if switch ~= nil then
                u:SetScriptBit('RULEUTC_ShieldToggle', switch)
            end
            n = n + 1
        end
    end
end
return n
"""
SHIELD_EVENTS = {20: (0.4, "nil", 1), 36: (1.5, "nil", 3), 44: (0, "false", 2), 60: (0, "true", 2)}

TOGGLES = r"""
local n = 0
for _, u in ArmyBrains[1]:GetListOfUnits(categories.ALLUNITS, false) do
    if EntityCategoryContains(categories.RADAR + categories.SHIELD, u) and u:GetFractionComplete() >= 1 then
        u:SetScriptBit('RULEUTC_IntelToggle', true)
        u:SetScriptBit('RULEUTC_ShieldToggle', true)
        n = n + 1
    end
end
return n
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"C:\faf\_enginetest\bin_c\main.exe")
    ap.add_argument("--script", default=str(HERE.parent / "ai" / "agentbridge.lua"))
    ap.add_argument("--steps", type=int, default=80)
    a = ap.parse_args()
    exe = Path(a.exe)
    env = FAEnv(GameConfig(map="SCMP_016", faction=1, ai="easy", ai_faction=2, seed=5, window=(640, 480),
                           minimized=True, exe=exe, bin_dir=exe.parent, script=Path(a.script)),
                step_ticks=10, privileged=True)
    coverage, problems, checked = collections.Counter(), [], 0
    shield_states = collections.Counter()
    try:
        env.reset()
        env.server.send({"config": {"allowEval": True, "obsCheck": True}, "queries": [{"type": "eval", "code": SPAWN}],
                         "wait": 1})
        msg = env._recv()
        print("spawn:", (msg.get("result") or {}).get("queries"), msg.get("errors"), flush=True)
        for i in range(a.steps):
            queries = [{"type": "eval", "code": TOGGLES}] if i in (30, 31) else []
            if i in SHIELD_EVENTS:
                queries.append({"type": "eval", "code": SHIELD_EVENT % SHIELD_EVENTS[i]})
            _, _, done, _, info = env.step(queries=queries, wait=10)
            if queries:
                print(f"step {i}:", [q.get("value", q.get("error")) for q in info.get("queries") or []], flush=True)
            if info.get("errors"):
                print("errors:", info["errors"][:3], flush=True)
            obs = env.last_obs or {}
            if "units_native" not in obs:
                problems.append(f"step {i}: no native list")
                continue
            checked += 1
            p = diff_units(obs.get("units") or [], obs.get("units_native") or [])
            if not same(obs.get("privileged"), obs.get("privileged_native"), 0):
                p.append("privileged differs")
            exact = json.dumps(obs.get("units"), sort_keys=True) == json.dumps(obs.get("units_native"), sort_keys=True)
            if not exact:
                p.append("not byte-identical after parsing")
            if p:
                problems.append(f"tick {obs.get('tick')}: {p[:5]}")
            for u in obs.get("units_native") or []:
                coverage.update(k for k in u if k not in ("id", "army", "x", "y", "z"))
                if "shield" in u:           # which shield states were seen: down/off, damaged, full
                    v = u["shield"]
                    shield_states[(u.get("bp"), "0" if v == 0 else "1" if v == 1 else "part")] += 1
            if done:
                break
    finally:
        env.close()
    print(f"checked {checked} observations; differences: {len(problems)}")
    for p in problems[:20]:
        print("  ", p)
    print("fields covered:", dict(coverage.most_common()))
    print("shield states seen (blueprint, down/part/full): observations",
          {f"{bp}:{state}": n for (bp, state), n in sorted(shield_states.items(), key=lambda kv: str(kv[0]))})
    sys.exit(1 if problems or not checked else 0)


if __name__ == "__main__":
    main()
