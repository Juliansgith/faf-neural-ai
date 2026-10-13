"""Build a trace-only Lua override folder for comparing the two engines' sim Lua.

    make_lua_trace.py [--out DIR] [--guap-ticks A:B]

Writes <out>/lua/simInit.lua: FAF's deployed simInit.lua (from gamedata/lua.nx2, so it matches the
running game exactly) with a tracer prepended, which runs before any module captures a local copy:

- every sim Random() call is logged as `RND <tick> <caller> n=<args> <result>` (plus the entity id and
  blueprint when the caller is a tree's method, via debug.getlocal);
- every DamageRing call is bracketed by `DRB` / `DRE` lines;
- with --guap-ticks, AI brains' GetUnitsAroundPoint results during those ticks are logged as `GUAP`.

The tracer only reads state, so it does not change the simulation (checked: the random-stream
fingerprints of traced and untraced runs are identical). Mount the folder with probe_run.py --mount and
diff two logs with rnd_diff.py. FA's sim errors on reads of undefined globals: keep tracer state local.
"""
import argparse
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FA_FIDELITY_DIR", REPO.parent / "_fidelity"))
NX2 = Path(r"C:\ProgramData\FAForever\gamedata\lua.nx2")

TRACER = r'''-- [fidelity trace] prepended by faf-neural-ai/tools/fidelity/make_lua_trace.py
do
    local rawRandom = Random
    local getinfo = debug and debug.getinfo
    local getlocal = debug and debug.getlocal
    local format, tostring, unpack, LOG = string.format, tostring, unpack, LOG
    Random = function(...)
        local r = rawRandom(unpack(arg))
        local where = "?"
        if getinfo then
            local i = getinfo(2, "Sl")
            if i then where = (i.short_src or "?") .. ":" .. tostring(i.currentline) end
        end
        local tick = GetGameTick and GetGameTick() or -1
        local extra = ""
        if getlocal and string.find(where, "proptree", 1, true) then
            local _, self = getlocal(2, 1)
            if type(self) == "table" and self.GetEntityId then
                local pos = self.CachePosition or self:GetPosition()
                extra = format(" id=%s at=(%.2f,%.2f) bp=%s", tostring(self:GetEntityId()), pos[1], pos[3],
                    tostring(self.Blueprint and self.Blueprint.BlueprintId))
            end
        end
        LOG(format("RND %d %s n=%d %s%s", tick, where, arg.n or 0, tostring(r), extra))
        return r
    end

    local rawDamageRing = DamageRing
    DamageRing = function(...)
        local p = arg[2]
        local tick = GetGameTick and GetGameTick() or -1
        LOG(format("DRB %d pos=(%.3f,%.3f,%.3f) r=%s..%s type=%s", tick, p[1], p[2], p[3],
            tostring(arg[3]), tostring(arg[4]), tostring(arg[6])))
        local r = rawDamageRing(unpack(arg))
        LOG(format("DRE %d", tick))
        return r
    end
@GUAP@
end
'''

GUAP = r'''
    local rawGUAP = moho.aibrain_methods.GetUnitsAroundPoint
    moho.aibrain_methods.GetUnitsAroundPoint = function(...)
        local r = rawGUAP(unpack(arg))
        local n = r and table.getn(r) or 0
        local tick = GetGameTick and GetGameTick() or -1
        if n > 0 and tick >= @LO@ and tick <= @HI@ then
            local p = arg[3]
            local ids = {}
            for _, u in r do
                local up = u:GetPosition()
                table.insert(ids, format("%s:%s@(%.2f,%.2f)", tostring(u:GetEntityId()),
                    u:GetBlueprint().BlueprintId, up[1], up[3]))
            end
            LOG(format("GUAP %d at=(%.2f,%.2f) r=%s %s -> %d %s", tick, p[1], p[3], tostring(arg[4]),
                tostring(arg[5]), n, table.concat(ids, " ")))
        end
        return r
    end
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "lua_trace"))
    ap.add_argument("--guap-ticks", help="log brain:GetUnitsAroundPoint results during ticks A:B")
    a = ap.parse_args()

    with zipfile.ZipFile(NX2) as z:
        name = next(n for n in z.namelist() if n.lower() == "lua/siminit.lua")
        original = z.read(name).decode("utf-8", errors="replace")
    tracer = TRACER
    if a.guap_ticks:
        lo, hi = a.guap_ticks.split(":")
        tracer = tracer.replace("@GUAP@", GUAP.replace("@LO@", lo).replace("@HI@", hi))
    else:
        tracer = tracer.replace("@GUAP@", "")
    dest = Path(a.out) / "lua" / "simInit.lua"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tracer + original, encoding="utf-8")
    print(f"wrote {dest} ({len(original)} bytes of FAF's simInit.lua after the tracer)")


if __name__ == "__main__":
    main()
