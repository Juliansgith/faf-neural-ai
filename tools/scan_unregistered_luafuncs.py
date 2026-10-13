"""Find Lua bindings the engine defines but never registers.

A binding is a `func_X_LuaFuncDef()` whose body builds a `CScrLuaBinder`; it only takes effect if some
startup path calls it (directly, or through a `register_X` thunk that is itself called). Lists every
definition with no call site outside its own definition and outside uncalled thunks, with the Lua name
and the init set it would publish into.

usage: scan_unregistered_luafuncs.py [<faf-re checkout>]   (default: faf-re-dev next to this repository)
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

CHECKOUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "faf-re-dev"
SDK = CHECKOUT / "src" / "sdk"
texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in SDK.rglob("*.cpp")}

defn_re = re.compile(r"CScrLuaInitForm\s*\*\s*(?:moho::)?(func_\w+_LuaFuncDef)\s*\(\s*\)\s*\{")
binder_re = re.compile(r"CScrLuaBinder\s+\w+\s*\(\s*([^,]+),\s*\"([^\"]+)\"\s*,[^;]*?\"([^\"]*)\"", re.S)
thunk_re = re.compile(r"(?:void|CScrLuaInitForm\s*\*)\s+(?:moho::)?(register_\w+)\s*\(\s*\)\s*\{([^}]*)\}", re.S)

definitions = {}          # func name -> (path, lua name, class, init set)
for path, text in texts.items():
    for m in defn_re.finditer(text):
        body = text[m.end(): m.end() + 1500]
        b = binder_re.search(body)
        definitions[m.group(1)] = (path, b.group(2) if b else "?", b.group(3) if b else "?",
                                   b.group(1).strip() if b else "?")

# register_X thunks: which func each forwards to
thunk_targets = defaultdict(set)
for path, text in texts.items():
    for m in thunk_re.finditer(text):
        for f in re.findall(r"(func_\w+_LuaFuncDef)", m.group(2)):
            thunk_targets[f].add(m.group(1))


def called(name: str, exclude_defs: bool = True) -> bool:
    """True if `name(` appears anywhere other than its own definition or a declaration."""
    pat = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    for text in texts.values():
        for m in pat.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start: text.find("\n", m.start())]
            if re.search(r"(CScrLuaInitForm\s*\*|void)\s+(moho::)?" + re.escape(name), line):
                continue   # a definition or declaration
            if line.strip().startswith(("*", "//")):
                continue   # comment
            return True
    return False


unregistered = []
for func, (path, lua_name, cls, init_set) in sorted(definitions.items()):
    direct = called(func)
    # called only from thunks? then the thunk must be called
    thunks = thunk_targets.get(func, set())
    reachable = False
    if direct:
        # direct call might be the thunk body itself; check whether any caller is not a thunk
        reachable = True
        if thunks and not any(called(t) for t in thunks):
            # every call site may be inside an uncalled thunk; verify by removing thunk bodies
            pat = re.compile(r"\b" + re.escape(func) + r"\s*\(")
            outside = 0
            for text in texts.values():
                stripped = thunk_re.sub("", text)
                for m in pat.finditer(stripped):
                    line_start = stripped.rfind("\n", 0, m.start()) + 1
                    line = stripped[line_start: stripped.find("\n", m.start())]
                    if re.search(r"(CScrLuaInitForm\s*\*|void)\s+(moho::)?" + re.escape(func), line):
                        continue
                    if line.strip().startswith(("*", "//")):
                        continue
                    outside += 1
            reachable = outside > 0
    if not reachable:
        unregistered.append((cls, lua_name, init_set, func, path.relative_to(SDK)))

print(f"{len(definitions)} LuaFuncDef definitions, {len(unregistered)} never called:")
for cls, lua_name, init_set, func, path in sorted(unregistered):
    print(f"  {cls:<28} {lua_name:<40} {init_set:<28} {func}  [{path}]")
