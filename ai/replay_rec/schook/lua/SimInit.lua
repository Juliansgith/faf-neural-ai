-- replay_rec: hooked onto /lua/SimInit.lua when FAF's own exe plays a replay for recording (fa_train.replay_rec).
--
-- For every human army it loads its own copy of the agent's observation script (/agentai/agentbridge.lua, the
-- file the agent bridge uses in our engine) and observes the game from that army's side every DECISION_TICKS
-- ticks, the way a training game is observed. Nothing here gives an order or touches the game state, so the replay
-- stays in sync (the sim keeps checking the players' checksums). The sim Lua has no file access: records
-- (kind|army|tick|seq|json) go to the UI through the sync table, and from there into the preferences file.

local DECISION_TICKS = 20

local Format = string.format
local Gsub = string.gsub
local TableInsert = table.insert
local TableConcat = table.concat

-- (an apostrophe too: the preferences file quotes strings with it)
local Escapes = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t',
                  ["'"] = '\\u0027' }
local Special = "[%c\"\\']"
local function EscapeChar(c)
    return Escapes[c] or Format('\\u%04x', string.byte(c))
end

local function Encode(v, out)
    local t = type(v)
    if t == 'table' then
        if v[1] ~= nil or next(v) == nil then
            TableInsert(out, '[')
            local n = table.getn(v)
            for i = 1, n do
                if i > 1 then TableInsert(out, ',') end
                Encode(v[i], out)
            end
            TableInsert(out, ']')
        else
            TableInsert(out, '{')
            local first = true
            for k, x in v do
                local kt = type(k)
                if (kt == 'string' or kt == 'number') and type(x) ~= 'function' and type(x) ~= 'cfunction' then
                    if not first then TableInsert(out, ',') end
                    first = false
                    TableInsert(out, '"' .. Gsub(tostring(k), Special, EscapeChar) .. '":')
                    Encode(x, out)
                end
            end
            TableInsert(out, '}')
        end
    elseif t == 'string' then
        TableInsert(out, '"' .. Gsub(v, Special, EscapeChar) .. '"')
    elseif t == 'number' then
        if v ~= v or v > 1e38 or v < -1e38 then          -- NaN, infinity (Lua 5.0 has no math.huge)
            TableInsert(out, 'null')
        elseif v == math.floor(v) and v > -1e9 and v < 1e9 then
            TableInsert(out, Format('%d', v))
        else
            TableInsert(out, Format('%.7g', v))
        end
    elseif t == 'boolean' then
        TableInsert(out, v and 'true' or 'false')
    else
        TableInsert(out, 'null')
    end
end

local Seq = 0
-- Hands a record to the UI side (the sync table goes to the user Lua every beat; the hook on /lua/UserSync.lua
-- stores it in the preferences file, which the recorder process reads). Not the game log: the engine keeps every
-- logged byte in its log window, and a few MB of observations exhaust the 32-bit address space.
local function Emit(kind, army, tick, value)
    local out = {}
    Encode(value, out)
    Seq = Seq + 1
    Sync.ReplayRec = Sync.ReplayRec or {}
    TableInsert(Sync.ReplayRec, Format('%s|%d|%d|%d|', kind, army, tick, Seq) .. TableConcat(out))
end

local function Recorder()
    local observers = {}
    for index, brain in ipairs(ArmyBrains) do
        if brain.BrainType == 'Human' and not ArmyIsCivilian(index) then
            local env = setmetatable({}, { __index = _G })
            local ok, err = pcall(doscript, '/agentai/agentbridge.lua', env)
            if not ok then
                WARN('REPLAYREC: the observation script did not load: ' .. tostring(err))
                return
            end
            env.AgentBridge_Act({ config = { armies = { index }, privileged = true, omniscient = false,
                                             heightmapSize = 64 } })
            TableInsert(observers, { army = index, env = env })
        end
    end
    for _, o in observers do
        local ok, hello = pcall(o.env.AgentBridge_Hello)
        if ok then
            Emit('H', o.army, GetGameTick(), hello)
        else
            WARN('REPLAYREC: hello failed: ' .. tostring(hello))
        end
    end
    local nextTick = 0
    while true do
        local tick = GetGameTick()
        if tick < nextTick then
            WaitTicks(nextTick - tick)
        else
            for _, o in observers do
                local ok, obs = pcall(o.env.AgentBridge_Observe)
                if ok then
                    Emit('O', o.army, tick, obs)
                else
                    WARN('REPLAYREC: observe failed: ' .. tostring(obs))
                end
            end
            nextTick = tick + DECISION_TICKS
            if math.mod(tick, 200) < DECISION_TICKS then
                local kb, limit = gcinfo()
                LOG(Format('REPLAYMEM %d lua %d KB limit %d KB', tick, kb, limit))
            end
            WaitTicks(1)
        end
    end
end

local _BeginSession = BeginSession
function BeginSession()
    _BeginSession()
    ForkThread(Recorder)
end
