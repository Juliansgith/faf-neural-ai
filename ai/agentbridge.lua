-- agentbridge.lua -- the sim-side half of the agent bridge (engine side: src/sdk/moho/agent/AgentBridge.cpp).
--
-- The engine loads this file into the sim Lua state when the game runs with
--   /agentbridge <port> /agentscript <this file>
-- and calls three globals:
--   AgentBridge_Hello()     static game information, sent once in the "hello" message
--   AgentBridge_Observe()   the observation, sent at every agent step
--   AgentBridge_Act(msg)    applies the agent's reply; the return value comes back as "result"
--
-- It is ordinary FA sim Lua: change it to change what the agent sees and what it can do. Actions
-- are limited to what a player could order; nothing here spawns units or grants resources.
--
-- Lua 5.0 (no '#', no 'continue'); numbers are single-precision floats; entity ids are strings.

local Bridge = {
    armies = { 1 },       -- army indices the agent commands; the first is "me"
    omniscient = false,   -- false: enemy units are visible only through the agent's own intel
    heightmapSize = 64,   -- samples per axis in the hello heightmap
    privileged = false,   -- true: add obs.privileged, every army's true totals (for critics and rewards only)
    ghosts = true,        -- include enemy structures seen before but not now (as a player remembers them)
    profile = 0,          -- 1: time the observation's sections (AgentBridge_Lap), 2: also per-unit parts
    nativeObs = true,     -- build the unit list in the engine (AgentBridge_ObserveUnits) when it can
    obsCheck = false,     -- both: the Lua list in obs.units, the native one in obs.units_native (testing)
    views = {},           -- more armies observed each step (self-play: obs.views["2"], army 2's own view)
}

local TableInsert = table.insert
local TableGetn = table.getn

-- Section timer provided by the engine (profiling only; absent in older engines).
local EngineLap = rawget(_G, 'AgentBridge_Lap')
local function Lap(name)
    if EngineLap then EngineLap(name) end
end

local function Contains(list, value)
    for _, v in ipairs(list) do
        if v == value then
            return true
        end
    end
    return false
end

local function Me()
    return Bridge.armies[1]
end

-- Protected call that returns the value or a fallback, for engine calls that may not apply. The
-- arguments are forwarded exactly: engine functions check their count, and a trailing nil counts.
local function Try(fallback, fn, ...)
    local ok, value = pcall(fn, unpack(arg))
    if ok then
        return value
    end
    return fallback
end

local function StripLoc(text)
    if type(text) ~= 'string' then
        return text
    end
    local _, finish = string.find(text, '^<LOC [^>]*>')
    if finish then
        return string.sub(text, finish + 1)
    end
    return text
end

---------------------------------------------------------------------------------------------------
-- Static information (hello)

local function MapInfo()
    local size = ScenarioInfo.size or { 512, 512 }
    local width, height = size[1], size[2]
    local n = Bridge.heightmapSize
    local terrain, surface = {}, {}
    for row = 0, n - 1 do
        local z = (row + 0.5) * height / n
        for col = 0, n - 1 do
            local x = (col + 0.5) * width / n
            TableInsert(terrain, GetTerrainHeight(x, z))
            TableInsert(surface, GetSurfaceHeight(x, z))
        end
    end

    local deposits = {}
    local MarkerUtilities = import('/lua/sim/markerutilities.lua')
    for _, kind in ipairs({ 'Mass', 'Hydrocarbon' }) do
        local markers = Try({}, MarkerUtilities.GetMarkersByType, kind)
        for _, marker in pairs(markers or {}) do
            local p = marker.position or marker.Position
            if p then
                TableInsert(deposits, { type = (kind == 'Mass') and 'mass' or 'hydro', x = p[1], z = p[3] })
            end
        end
    end

    local playable = ScenarioInfo.PlayableArea
    return {
        name = ScenarioInfo.name,
        file = ScenarioInfo.file,
        width = width,
        height = height,
        playable = playable and { playable[1], playable[2], playable[3], playable[4] } or { 0, 0, width, height },
        heightmap = { n = n, terrain = terrain, surface = surface },
        deposits = deposits,
    }
end

local function TechLevel(categories)
    for _, c in ipairs(categories) do
        if c == 'TECH1' then return 1 end
        if c == 'TECH2' then return 2 end
        if c == 'TECH3' then return 3 end
        if c == 'EXPERIMENTAL' then return 4 end
    end
    return 0
end

local function Buildable(bp)
    local expressions = bp.Economy and bp.Economy.BuildableCategory
    if not expressions then
        return nil
    end
    local set, list = {}, {}
    for _, expression in ipairs(expressions) do
        local ids = Try({}, function(e) return EntityCategoryGetUnitList(ParseEntityCategory(e)) end, expression)
        for _, id in ipairs(ids or {}) do
            if not set[id] then
                set[id] = true
                TableInsert(list, id)
            end
        end
    end
    return list
end

-- Buildable lists per blueprint id, computed once (category parsing is not cheap).
local BuildableCache = {}
local function BuildableOf(id, bp)
    if BuildableCache[id] == nil then
        BuildableCache[id] = Buildable(bp) or false
    end
    return BuildableCache[id] or nil
end

-- A commander's or support commander's enhancements: slot, cost, and what each needs first. The "...Remove"
-- entries (taking an enhancement off again) are left out.
local function Enhancements(bp)
    if type(bp.Enhancements) ~= 'table' then
        return nil
    end
    local out = nil
    for name, e in pairs(bp.Enhancements) do
        if type(e) == 'table' and e.Slot and not e.RemoveEnhancements then
            out = out or {}
            TableInsert(out, { name = name, slot = e.Slot, mass = e.BuildCostMass, energy = e.BuildCostEnergy,
                               time = e.BuildTime, requires = e.Prerequisite, health = e.NewHealth,
                               buildRate = e.NewBuildRate })
        end
    end
    return out
end

-- Abilities a player can switch (shield, stealth, jamming, production, ...), by their script bit.
local ToggleBits = {
    RULEUTC_ShieldToggle = 0, RULEUTC_WeaponToggle = 1, RULEUTC_JammingToggle = 2, RULEUTC_IntelToggle = 3,
    RULEUTC_ProductionToggle = 4, RULEUTC_StealthToggle = 5, RULEUTC_GenericToggle = 6, RULEUTC_SpecialToggle = 7,
    RULEUTC_CloakToggle = 8,
}
local ToggleNames = {}              -- bit -> name: the engine's Get/SetScriptBit take the name, as FAF's scripts do
for name, bit in pairs(ToggleBits) do ToggleNames[bit] = name end
-- Which of them a unit has: from the blueprint's ToggleCaps (a set { RULEUTC_x = true }, a list, or one name --
-- type-guarded, as not every blueprint uses the same form), else asked of the unit. The engine's TestToggleCaps
-- alone is not enough: in faf-re it tests the blueprint's command caps field (it answered no for a radar's
-- intel toggle). Cached per blueprint once a unit of it is complete.
local ToggleCache = {}
local function ToggleBitsOf(caps, found)
    if type(caps) == 'table' then
        for k, v in pairs(caps) do
            local name = (type(k) == 'string' and v and k) or (type(v) == 'string' and v) or nil
            if name and ToggleBits[name] then found[ToggleBits[name]] = true end
        end
    elseif type(caps) == 'string' and ToggleBits[caps] then
        found[ToggleBits[caps]] = true
    end
end
local function UnitToggles(unit)
    local id = unit.UnitId
    if ToggleCache[id] ~= nil then
        return ToggleCache[id] or nil
    end
    local found = {}
    local general = unit:GetBlueprint().General
    Try(nil, ToggleBitsOf, general and general.ToggleCaps, found)
    if next(found) == nil then
        for cap, bit in pairs(ToggleBits) do
            if Try(false, unit.TestToggleCaps, unit, cap) then found[bit] = true end
        end
    end
    local bits = nil
    for bit = 0, 8 do
        if found[bit] then
            bits = bits or {}
            TableInsert(bits, bit)
        end
    end
    if unit:GetFractionComplete() >= 1 then
        ToggleCache[id] = bits or false
    end
    return bits
end

local function BlueprintInfo(id, bp)
    local economy = bp.Economy or {}
    local physics = bp.Physics or {}
    local general = bp.General or {}
    local weapons = {}
    for _, w in ipairs(bp.Weapon or {}) do
        TableInsert(weapons, {
            label = w.Label,
            category = w.WeaponCategory,
            range = w.MaxRadius,
            damage = w.Damage,
            rof = w.RateOfFire,
            rangeCategory = w.RangeCategory,
        })
    end
    local categories = bp.Categories or {}
    return {
        id = id,
        name = StripLoc(general.UnitName) or StripLoc(bp.Description),
        description = StripLoc(bp.Description),
        faction = general.FactionName,
        tech = TechLevel(categories),
        categories = categories,
        mass = economy.BuildCostMass,
        energy = economy.BuildCostEnergy,
        buildTime = economy.BuildTime,
        buildRate = economy.BuildRate,
        massProduction = economy.ProductionPerSecondMass,
        energyProduction = economy.ProductionPerSecondEnergy,
        energyUpkeep = economy.MaintenanceConsumptionPerSecondEnergy,
        massStorage = economy.StorageMass,
        energyStorage = economy.StorageEnergy,
        health = bp.Defense and bp.Defense.MaxHealth,
        shieldSize = bp.Defense and bp.Defense.Shield and bp.Defense.Shield.ShieldSize,
        speed = physics.MaxSpeed,
        motion = physics.MotionType,
        footprint = bp.Footprint and { bp.Footprint.SizeX, bp.Footprint.SizeZ } or nil,
        -- the build skirt decides adjacency (Unit::OverlapsWith): { sizeX, sizeZ, offsetX, offsetZ }
        skirt = { physics.SkirtSizeX or 0, physics.SkirtSizeZ or 0, physics.SkirtOffsetX or 0,
                  physics.SkirtOffsetZ or 0 },
        vision = bp.Intel and bp.Intel.VisionRadius,
        radar = bp.Intel and bp.Intel.RadarRadius,
        weapons = weapons,
        upgradesTo = general.UpgradesTo ~= '' and general.UpgradesTo or nil,
        buildable = BuildableOf(id, bp),
        enhancements = Enhancements(bp),
    }
end

local function UnitBlueprints()
    local list = {}
    for id, bp in pairs(__blueprints) do
        if type(id) == 'string' and bp.Categories and bp.Defense and bp.Economy and bp.Physics and bp.General
            and bp.BlueprintId == id then
            local ok, info = pcall(BlueprintInfo, id, bp)
            if ok then
                TableInsert(list, info)
            end
        end
    end
    return list
end

local function Relation(me, other)
    if me == other then return 'self' end
    if IsAlly(me, other) then return 'ally' end
    if IsEnemy(me, other) then return 'enemy' end
    return 'neutral'
end

function AgentBridge_Hello()
    local me = Me()
    local armies = {}
    for index, brain in ipairs(ArmyBrains) do
        local x, z = brain:GetArmyStartPos()
        TableInsert(armies, {
            index = index,
            name = brain.Name,
            nickname = brain.Nickname,
            faction = brain:GetFactionIndex(),
            brainType = brain.BrainType,
            civilian = ArmyIsCivilian(index),
            relation = Relation(me, index),
            start = { x, z },
        })
    end
    return {
        map = MapInfo(),
        armies = armies,
        blueprints = UnitBlueprints(),
        config = { armies = Bridge.armies, omniscient = Bridge.omniscient },
    }
end

---------------------------------------------------------------------------------------------------
-- Observation

local StatNames = {
    'Enemies_Killed', 'Units_Killed', 'Units_History',
    'Enemies_MassValue_Destroyed', 'Enemies_EnergyValue_Destroyed',
    'Units_MassValue_Lost', 'Units_EnergyValue_Lost',
    'Units_MassValue_Built', 'Units_EnergyValue_Built',
    'Economy_TotalProduced_Mass', 'Economy_TotalProduced_Energy',
    'Economy_TotalConsumed_Mass', 'Economy_TotalConsumed_Energy',
    'Economy_Reclaimed_Mass', 'Economy_Reclaimed_Energy',
    'Enemies_Commanders_Destroyed',
}

local function Economy(brain)
    local result = {}
    for _, resource in ipairs({ 'MASS', 'ENERGY' }) do
        local stored = brain:GetEconomyStored(resource)
        local ratio = brain:GetEconomyStoredRatio(resource)
        -- The engine reports rates per tick; these are per second.
        result[string.lower(resource)] = {
            stored = stored,
            max = (ratio > 0) and (stored / ratio) or nil,
            income = brain:GetEconomyIncome(resource) * 10,
            requested = brain:GetEconomyRequested(resource) * 10,
            usage = (Try(0, brain.GetEconomyUsage, brain, resource) or 0) * 10,
            trend = brain:GetEconomyTrend(resource) * 10,
        }
    end
    return result
end

local function Stats(brain)
    local result = {}
    for _, name in ipairs(StatNames) do
        local stat = Try(nil, brain.GetArmyStat, brain, name, 0)
        result[name] = stat and stat.Value or 0
    end
    return result
end

local function BaseDescription(unit, x, y, z)
    return { id = unit.EntityId, army = unit.Army, x = x, y = y, z = z }
end

local function DescribeVisibleUnit(unit)
    local x, y, z = unit:GetPositionXYZ()
    local d = BaseDescription(unit, x, y, z)
    d.bp = unit.UnitId
    d.hp = unit:GetHealth()
    d.maxhp = unit:GetMaxHealth()
    d.frac = unit:GetFractionComplete()
    d.layer = unit.Layer or unit:GetCurrentLayer()
    d.heading = unit:GetHeading()
    return d
end

-- A unit's own shield (FAF sim: unit.MyShield, a personal or bubble shield entity): its health over its maximum
-- while it is up (Shield:IsUp -- switched on and enabled, not recharging or drained), 0 while it is down or off,
-- nil when the unit has none. (Unit:GetShieldRatio is the UI's shield bar, which also shows the recharge
-- progress.) The native unit list (AgentObserve.cpp) computes exactly this.
local function ShieldValue(unit)
    local shield = unit.MyShield
    if type(shield) ~= 'table' then
        return nil
    end
    local health = Try(nil, shield.GetHealth, shield)
    local max = Try(nil, shield.GetMaxHealth, shield)
    if not health or not max then
        return nil                      -- the shield entity is gone
    end
    if not Try(false, shield.IsUp, shield) or max <= 0 then
        return 0
    end
    return health / max
end

local function DescribeOwnUnit(unit)
    local fine = Bridge.profile >= 2
    if fine then Lap('own_loop') end
    local d = DescribeVisibleUnit(unit)
    if fine then Lap('own_base') end
    d.idle = unit:IsIdleState()
    d.work = unit:GetWorkProgress()
    local queue = unit:GetCommandQueue() or {}
    d.queue = TableGetn(queue)
    -- what the unit is doing right now, as a player sees it: the current order and where it leads
    local first = queue[1]
    if first then
        -- our engine's fields are CommandType/X/Z, FAF's exe (replay recording) names them commandType/x/z
        d.order = first.CommandType or first.commandType
        local x, z = first.X or first.x, first.Z or first.z
        if x and z and (x ~= 0 or z ~= 0) then
            d.dest = { x, z }
        end
    end
    local focus = unit:GetFocusUnit()
    if focus and focus.EntityId then
        d.focus = focus.EntityId
    end
    if unit:IsPaused() then
        d.paused = true
    end
    d.shield = ShieldValue(unit)
    if fine then Lap('own_state') end
    -- what a structure actually yields and costs right now: adjacency bonuses show up here
    if EntityCategoryContains(categories.STRUCTURE, unit) then
        local pm = Try(0, unit.GetProductionPerSecondMass, unit) or 0
        local pe = Try(0, unit.GetProductionPerSecondEnergy, unit) or 0
        local ce = Try(0, unit.GetConsumptionPerSecondEnergy, unit) or 0
        if pm > 0 then d.pm = pm end
        if pe > 0 then d.pe = pe end
        if ce > 0 then d.ce = ce end
    end
    if fine then Lap('own_prod') end
    -- builders: what they may build right now, as the engine rules it (tech, the HQ rule for support
    -- factories, army restrictions, the unit cap)
    -- (two tests, not the union ENGINEER + FACTORY: faf-re's category union dropped members that sort before
    -- the left set's first one -- every Aeon factory, 2026-09-30)
    local factory = EntityCategoryContains(categories.FACTORY, unit)
    if (EntityCategoryContains(categories.ENGINEER, unit) or factory)
            and (unit:GetFractionComplete() >= 1 or (factory and unit:GetFractionComplete() >= 0.9)) then
        local list = BuildableOf(unit.UnitId, unit:GetBlueprint())
        if list then
            local can = {}
            for _, id in ipairs(list) do
                if Try(false, unit.CanBuild, unit, id) then
                    TableInsert(can, id)
                end
            end
            d.can_build = can
        end
    end
    if fine then Lap('own_canbuild') end
    -- commander and support commander enhancements installed
    local enh = SimUnitEnhancements and SimUnitEnhancements[unit.EntityId]
    if enh then
        local list = {}
        for _, name in pairs(enh) do
            TableInsert(list, name)
        end
        if TableGetn(list) > 0 then d.enh = list end
    end
    -- missiles ready in a silo, units carried, fire state, abilities switched on
    local tml = Try(0, unit.GetTacticalSiloAmmoCount, unit)
    if tml and tml > 0 then d.tml = tml end
    local nukes = Try(0, unit.GetNukeSiloAmmoCount, unit)
    if nukes and nukes > 0 then d.nukes = nukes end
    if EntityCategoryContains(categories.TRANSPORTATION, unit) then
        local cargo = Try(nil, unit.GetCargo, unit)
        d.cargo = cargo and TableGetn(cargo) or 0
    end
    local fire = Try(0, unit.GetFireState, unit)
    if fire and fire ~= 0 then d.fire = fire end
    if fine then Lap('own_misc') end
    local bits = UnitToggles(unit)
    if bits then
        d.toggles = bits
        local on = {}
        for _, bit in ipairs(bits) do
            if Try(false, unit.GetScriptBit, unit, ToggleNames[bit]) then TableInsert(on, bit) end
        end
        if TableGetn(on) > 0 then d.toggles_on = on end
    end
    if fine then Lap('own_toggles') end
    return d
end

-- 'seen', 'radar' or nil, from army `me`'s point of view.
local function IntelOn(unit, me)
    local blip = unit:GetBlip(me)
    if not blip then
        return nil, nil
    end
    if blip:IsSeenNow(me) then
        return 'seen', blip
    end
    if blip:IsOnRadar(me) or blip:IsOnSonar(me) or blip:IsOnOmni(me) then
        return 'radar', blip
    end
    return nil, blip
end

local function IsLive(unit)
    return unit and not unit.Dead and not unit:BeenDestroyed()
end

-- Per-blueprint facts the privileged summary needs, computed once per blueprint id.
local UnitKinds = {}
local function UnitKind(id)
    local kind = UnitKinds[id]
    if kind then
        return kind
    end
    local bp = __blueprints[id] or {}
    local cats = {}
    for _, c in ipairs(bp.Categories or {}) do
        cats[c] = true
    end
    kind = {
        mass = (bp.Economy and bp.Economy.BuildCostMass) or 0,
        command = cats.COMMAND or false,
        engineer = (cats.ENGINEER and not cats.COMMAND) or false,
        structure = cats.STRUCTURE or false,
        factory = (cats.FACTORY and cats.STRUCTURE) or false,
        mex = cats.MASSEXTRACTION or false,
        combat = (cats.MOBILE and not cats.ENGINEER and not cats.COMMAND
            and bp.Weapon and TableGetn(bp.Weapon) > 0) or false,
    }
    UnitKinds[id] = kind
    return kind
end

-- True totals for one army: what a critic or a reward may use, never the policy.
local function PrivilegedSummary(brain, units)
    local s = {
        massIncome = brain:GetEconomyIncome('MASS') * 10, energyIncome = brain:GetEconomyIncome('ENERGY') * 10,
        massStored = brain:GetEconomyStored('MASS'), energyStored = brain:GetEconomyStored('ENERGY'),
        armyMass = 0, structureMass = 0, engineerMass = 0,
        combat = 0, engineers = 0, factories = 0, mexes = 0, structures = 0, commanderHealth = 0,
    }
    for _, unit in ipairs(units) do
        if IsLive(unit) then
            local kind = UnitKind(unit.UnitId)
            local value = kind.mass * unit:GetFractionComplete()
            if kind.command then
                s.commanderHealth = unit:GetHealth() / math.max(1, unit:GetMaxHealth())
            elseif kind.engineer then
                s.engineers = s.engineers + 1
                s.engineerMass = s.engineerMass + value
            elseif kind.combat then
                s.combat = s.combat + 1
                s.armyMass = s.armyMass + value
            elseif kind.structure then
                s.structures = s.structures + 1
                s.structureMass = s.structureMass + value
                if kind.factory then s.factories = s.factories + 1 end
                if kind.mex then s.mexes = s.mexes + 1 end
            end
        end
    end
    return s
end

-- One army's units as the agent may know them, appended to `units` (the Lua build of the unit list; the
-- engine's AgentBridge_ObserveUnits builds the same list natively). Returns the army's unit list.
local function ObserveArmyUnits(units, me, brain, relation, known, profile)
    local list = brain:GetListOfUnits(categories.ALLUNITS, false) or {}
    if profile then Lap('unit_lists') end
    for _, unit in ipairs(list) do
        if IsLive(unit) then
            if relation == 'self' then
                TableInsert(units, DescribeOwnUnit(unit))
            elseif known then
                TableInsert(units, DescribeVisibleUnit(unit))
            else
                local intel, blip = IntelOn(unit, me)
                if intel == 'seen' then
                    TableInsert(units, DescribeVisibleUnit(unit))
                elseif intel == 'radar' then
                    -- Radar contacts: a (possibly jammed) position and nothing about the unit.
                    local p = blip:GetPosition()
                    local d = BaseDescription(unit, p[1], p[2], p[3])
                    d.radar = true
                    TableInsert(units, d)
                elseif blip and Bridge.ghosts and UnitKind(unit.UnitId).structure
                    and Try(false, blip.IsSeenEver, blip, me) then
                    -- A structure seen before: its type and place, as a player remembers it. (It
                    -- vanishes when destroyed, which a player would only learn by looking again.)
                    local p = blip:GetPosition()
                    local d = BaseDescription(unit, p[1], p[2], p[3])
                    d.bp = unit.UnitId
                    d.ghost = true
                    TableInsert(units, d)
                end
            end
        end
    end
    if profile then Lap((relation == 'self' and 'units_own') or (known and 'units_known') or 'units_intel') end
    return list
end

-- The engine's native unit list (moho/agent/AgentObserve.cpp): the same units and fields as
-- ObserveArmyUnits, and the unit part of PrivilegedSummary, without a Lua table per unit. Engines
-- without it (or a failure) use the Lua build.
local NativeObserveUnits = rawget(_G, 'AgentBridge_ObserveUnits')
local NativeCtx = nil
local function NativeContext(me)
    if not NativeCtx then
        NativeCtx = {
            brains = ArmyBrains, modes = {}, summary = {},
            cats = { ALLUNITS = categories.ALLUNITS, STRUCTURE = categories.STRUCTURE,
                     ENGINEER = categories.ENGINEER, FACTORY = categories.FACTORY,
                     TRANSPORTATION = categories.TRANSPORTATION },
            -- asked once per blueprint
            buildable = function(unit) return BuildableOf(unit.UnitId, unit:GetBlueprint()) end,
            toggles = UnitToggles,
            kind = function(unit) return UnitKind(unit.UnitId) end,
        }
    end
    NativeCtx.me = me
    NativeCtx.ghosts = Bridge.ghosts
    return NativeCtx
end

-- The privileged summary's economy part (PrivilegedSummary's first lines), added to a native summary.
local function AddPrivilegedEconomy(brain, s)
    s.massIncome = brain:GetEconomyIncome('MASS') * 10
    s.energyIncome = brain:GetEconomyIncome('ENERGY') * 10
    s.massStored = brain:GetEconomyStored('MASS')
    s.energyStored = brain:GetEconomyStored('ENERGY')
    return s
end

-- One army's observation. `native`: the engine's unit list may be used -- once per step only (it fills one
-- buffer that is encoded when the message is sent), so the further views of self-play use the Lua one.
local function ObserveFor(me, native)
    local ctx = (native and Bridge.nativeObs and NativeObserveUnits) and NativeContext(me) or nil
    local luaUnits = not ctx or Bridge.obsCheck
    local obs = {
        tick = GetGameTick(),
        time = GetGameTimeSeconds(),
        me = me,
        armies = {},
        units = {},
    }
    local units = obs.units
    local profile = Bridge.profile >= 1
    if profile then Lap('observe_setup') end

    for index, brain in ipairs(ArmyBrains) do
        local relation = Relation(me, index)
        local known = relation == 'self' or relation == 'ally' or Bridge.omniscient
        local army = {
            index = index,
            relation = relation,
            civilian = ArmyIsCivilian(index),
            defeated = brain:IsDefeated(),
            status = brain.Status,
        }
        if known then
            army.eco = Economy(brain)
            army.stats = Stats(brain)
            army.unitCap = GetArmyUnitCap(index)
            army.unitCount = GetArmyUnitCostTotal(index)
        end
        TableInsert(obs.armies, army)
        if profile then Lap('armies') end

        local summary = Bridge.privileged and not obs.armies[index].civilian
        if ctx then
            ctx.modes[index] = (relation == 'self' and 1) or (known and 2) or 3
            ctx.summary[index] = summary
        end
        if luaUnits then
            local list = ObserveArmyUnits(units, me, brain, relation, known, profile)
            if summary then
                obs.privileged = obs.privileged or {}
                obs.privileged[tostring(index)] = PrivilegedSummary(brain, list)
            end
            if profile then Lap('privileged') end
        end
    end

    if ctx then
        local token, summaries = NativeObserveUnits(ctx)
        if profile then Lap('units_native') end
        if token == nil then
            WARN('AgentBridge: the native unit list failed (' .. tostring(summaries) .. '); using the Lua one')
            Bridge.nativeObs = false
            if not luaUnits then
                return ObserveFor(me, false)
            end
        else
            local privileged = nil
            for index, brain in ipairs(ArmyBrains) do
                local s = summaries[index]
                if s then
                    privileged = privileged or {}
                    privileged[tostring(index)] = AddPrivilegedEconomy(brain, s)
                end
            end
            if luaUnits then            -- the check: both lists, compared by the agent
                obs.units_native = token
                obs.privileged_native = privileged
            else
                obs.units = token
                obs.privileged = privileged
            end
        end
    end
    return obs
end

function AgentBridge_Observe()
    local obs = ObserveFor(Me(), true)
    if TableGetn(Bridge.views) > 0 then
        obs.views = {}
        for _, index in ipairs(Bridge.views) do
            if index ~= Me() and ArmyBrains[index] then
                obs.views[tostring(index)] = ObserveFor(index, false)
            end
        end
    end
    if Bridge.capture and Bridge.capture.on then
        -- what another AI ordered the controlled armies since the last observation (each order names its army)
        obs.orders = Bridge.takeOrders()
    end
    return obs
end

---------------------------------------------------------------------------------------------------
-- Actions and queries

local function ActingArmy(action)
    local army = action.army or Me()
    if not Contains(Bridge.armies, army) then
        error('army ' .. tostring(army) .. ' is not controlled by the agent', 0)
    end
    return army
end

local function ResolveUnits(action)
    local army = ActingArmy(action)
    local units = {}
    for _, id in ipairs(action.units or {}) do
        local entity = GetEntityById(tostring(id))
        if entity and entity.UnitId and IsLive(entity) and entity.Army == army then
            TableInsert(units, entity)
        end
    end
    if TableGetn(units) == 0 then
        error('none of the given units exist and belong to army ' .. tostring(army), 0)
    end
    return units
end

local function Position(action)
    local p = action.pos
    local x, z
    if p then
        x, z = p[1], p[2]
    else
        x, z = action.x, action.z
    end
    if type(x) ~= 'number' or type(z) ~= 'number' then
        error('missing position (pos = [x, z])', 0)
    end
    return { x, GetSurfaceHeight(x, z), z }
end

-- An entity the acting army may target: its own or allied units, any prop or wreck, and enemy units
-- it currently sees or has on radar (unless omniscient).
local function TargetEntity(action)
    local entity = action.target and GetEntityById(tostring(action.target))
    if not entity or (entity.BeenDestroyed and entity:BeenDestroyed()) then
        error('unknown target ' .. tostring(action.target), 0)
    end
    local army = ActingArmy(action)
    if entity.UnitId and not Bridge.omniscient and not IsAlly(army, entity.Army) and entity.Army ~= army then
        local intel = IntelOn(entity, army)
        if not intel then
            error('target ' .. tostring(action.target) .. ' is not visible', 0)
        end
    end
    return entity
end

local function Target(action)
    if action.target then
        return TargetEntity(action)
    end
    return Position(action)
end

local function Blueprint(action)
    local id = action.bp
    if type(id) ~= 'string' or not __blueprints[id] then
        error('unknown blueprint ' .. tostring(id), 0)
    end
    return id
end

local Actions = {}

Actions.move = function(a) IssueMove(ResolveUnits(a), Position(a)) end
Actions.attack_move = function(a) IssueAggressiveMove(ResolveUnits(a), Position(a)) end
Actions.patrol = function(a) IssuePatrol(ResolveUnits(a), Position(a)) end
Actions.attack = function(a) IssueAttack(ResolveUnits(a), Target(a)) end
Actions.stop = function(a) IssueStop(ResolveUnits(a)) end
Actions.clear = function(a) IssueClearCommands(ResolveUnits(a)) end
Actions.guard = function(a) IssueGuard(ResolveUnits(a), Target(a)) end
Actions.repair = function(a) IssueRepair(ResolveUnits(a), TargetEntity(a)) end
Actions.reclaim = function(a) IssueReclaim(ResolveUnits(a), TargetEntity(a)) end
Actions.capture = function(a) IssueCapture(ResolveUnits(a), TargetEntity(a)) end
Actions.build = function(a) IssueBuildMobile(ResolveUnits(a), Position(a), Blueprint(a), {}) end
Actions.factory_build = function(a) IssueBuildFactory(ResolveUnits(a), Blueprint(a), a.count or 1) end
Actions.upgrade = function(a) IssueUpgrade(ResolveUnits(a), Blueprint(a)) end
Actions.rally = function(a) IssueFactoryRallyPoint(ResolveUnits(a), Position(a)) end
Actions.overcharge = function(a) IssueOverCharge(ResolveUnits(a), TargetEntity(a)) end
Actions.nuke = function(a) IssueNuke(ResolveUnits(a), Position(a)) end
Actions.tactical = function(a) IssueTactical(ResolveUnits(a), Target(a)) end
Actions.transport_load = function(a) IssueTransportLoad(ResolveUnits(a), TargetEntity(a)) end
Actions.transport_unload = function(a) IssueTransportUnload(ResolveUnits(a), Position(a)) end
Actions.dive = function(a) IssueDive(ResolveUnits(a)) end
Actions.self_destruct = function(a) IssueKillSelf(ResolveUnits(a)) end
Actions.enhance = function(a)
    IssueScript(ResolveUnits(a), { TaskName = 'EnhanceTask', Enhancement = a.name })
end
Actions.silo_build = function(a)
    local units = ResolveUnits(a)
    if a.nuke then IssueSiloBuildNuke(units) else IssueSiloBuildTactical(units) end
end
Actions.pause = function(a)
    for _, unit in ipairs(ResolveUnits(a)) do unit:SetPaused(a.on ~= false) end
end
Actions.fire_state = function(a)
    for _, unit in ipairs(ResolveUnits(a)) do unit:SetFireState(a.state) end
end
Actions.toggle = function(a)
    local name = ToggleNames[a.bit] or a.bit
    for _, unit in ipairs(ResolveUnits(a)) do unit:SetScriptBit(name, a.on ~= false) end
end

local Queries = {}

Queries.can_build = function(q)
    local brain = ArmyBrains[ActingArmy(q)]
    return brain:CanBuildStructureAt(Blueprint(q), Position(q)) and true or false
end

-- Structures sit on the build grid: odd footprints centre on half cells, even ones on whole cells.
local function Snap(value, size)
    if math.mod(size or 1, 2) == 1 then
        return math.floor(value) + 0.5
    end
    return math.floor(value + 0.5)
end

-- The skirt rectangle a structure of blueprint `bp` occupies at (x, z), as RUnitBlueprint::GetSkirtRect
-- computes it: x0, z0, x1, z1.
local function SkirtRect(bp, x, z)
    local footprint = bp.Footprint or {}
    local fx, fz = footprint.SizeX or 1, footprint.SizeZ or 1
    local physics = bp.Physics or {}
    local xl = math.floor(x - fx * 0.5 + 0.5)
    local zl = math.floor(z - fz * 0.5 + 0.5)
    local x0, x1, z0, z1
    if (physics.SkirtSizeX or 0) == 0 then
        x0, x1 = xl, xl + fx
    else
        x0 = xl + (physics.SkirtOffsetX or 0)
        x1 = x0 + physics.SkirtSizeX
    end
    if (physics.SkirtSizeZ or 0) == 0 then
        z0, z1 = zl, zl + fz
    else
        z0 = zl + (physics.SkirtOffsetZ or 0)
        z1 = z0 + physics.SkirtSizeZ
    end
    return x0, z0, x1, z1
end

-- Positions where a structure of blueprint id `id` would be adjacent to `anchor` (a structure), by the
-- engine's rule: the skirts share an edge and one's side lies within the other's. Sides W, E, N, S.
local function AdjacentCandidates(id, anchor)
    local bp = __blueprints[id]
    local ax, _, az = anchor:GetPositionXYZ()
    local a0x, a0z, a1x, a1z = SkirtRect(anchor:GetBlueprint(), ax, az)
    -- the new building's skirt relative to its own position (skirt edges at integer steps)
    local footprint = bp.Footprint or {}
    local fx, fz = footprint.SizeX or 1, footprint.SizeZ or 1
    local physics = bp.Physics or {}
    local sx = ((physics.SkirtSizeX or 0) ~= 0) and physics.SkirtSizeX or fx
    local sz = ((physics.SkirtSizeZ or 0) ~= 0) and physics.SkirtSizeZ or fz
    local ox = ((physics.SkirtSizeX or 0) ~= 0) and (physics.SkirtOffsetX or 0) or 0
    local oz = ((physics.SkirtSizeZ or 0) ~= 0) and (physics.SkirtOffsetZ or 0) or 0
    -- position from the skirt's low corner: x0 = floor(x - fx/2 + 0.5) + ox, so x = x0 - ox + fx/2
    local function at(x0, z0) return x0 - ox + fx * 0.5, z0 - oz + fz * 0.5 end
    local out = {}
    local function along(fixedLow, isX)
        -- slide along the side so that one skirt's extent contains the other's
        local lo, hi, size = (isX and a0z or a0x), (isX and a1z or a1x), (isX and sz or sx)
        local first, last = lo, hi - size
        if size > hi - lo then first, last = hi - size, lo end
        for s = first, last do
            if isX then TableInsert(out, { at(fixedLow, s) }) else TableInsert(out, { at(s, fixedLow) }) end
        end
    end
    along(a0x - sx, true)      -- west: new x1 = anchor x0
    along(a1x, true)           -- east: new x0 = anchor x1
    along(a0z - sz, false)     -- north: new z1 = anchor z0
    along(a1z, false)          -- south: new z0 = anchor z1
    return out
end

-- Nearest spot (spiralling out from pos, up to `radius`) where the acting army can place `bp`. With
-- `adjacent_to` (unit ids, in order of preference) spots touching one of those structures are tried
-- first -- adjacency bonuses -- and with `adjacent_only` nothing else is.
Queries.find_build_site = function(q)
    local brain = ArmyBrains[ActingArmy(q)]
    local id = Blueprint(q)
    if q.adjacent_to then
        for _, anchorId in ipairs(q.adjacent_to) do
            local anchor = GetEntityById(tostring(anchorId))
            if anchor and not anchor.Dead and anchor.GetBlueprint and EntityCategoryContains(categories.STRUCTURE, anchor) then
                local candidates = AdjacentCandidates(id, anchor)
                if q.pos then           -- the touching spots nearest the one asked for first
                    local c = Position(q)
                    table.sort(candidates, function(a, b)
                        return (a[1] - c[1]) ^ 2 + (a[2] - c[3]) ^ 2 < (b[1] - c[1]) ^ 2 + (b[2] - c[3]) ^ 2
                    end)
                end
                for _, p in ipairs(candidates) do
                    if brain:CanBuildStructureAt(id, { p[1], GetSurfaceHeight(p[1], p[2]), p[2] }) then
                        return { p[1], p[2] }
                    end
                end
            end
        end
        if q.adjacent_only then
            return nil
        end
    end
    local centre = Position(q)
    local footprint = __blueprints[id].Footprint or {}
    local sx, sz = footprint.SizeX or 1, footprint.SizeZ or 1
    local step = math.max(1, math.max(sx, sz))
    local radius = q.radius or 30
    for ring = 0, math.ceil(radius / step) do
        local r = ring * step
        for dx = -r, r, step do
            for dz = -r, r, step do
                if math.max(math.abs(dx), math.abs(dz)) == r then
                    local x, z = Snap(centre[1] + dx, sx), Snap(centre[3] + dz, sz)
                    if brain:CanBuildStructureAt(id, { x, GetSurfaceHeight(x, z), z }) then
                        return { x, z }
                    end
                end
            end
        end
    end
    return nil
end

-- Mass (type 1) or hydrocarbon (type 2) deposits around a point; type 0 for both.
Queries.deposits = function(q)
    local centre = Position(q)
    local found = GetDepositsAroundPoint(centre[1], centre[3], q.radius or 30, q.kind or 0) or {}
    local result = {}
    for _, d in ipairs(found) do
        TableInsert(result, {
            x = (d.X1 + d.X2) * 0.5, z = (d.Z1 + d.Z2) * 0.5, type = d.Type, dist = d.Dist,
        })
    end
    return result
end

-- Reclaimable props and wrecks in a square around a point.
Queries.reclaim = function(q)
    local centre = Position(q)
    local r = q.radius or 20
    local found = GetReclaimablesInRect(Rect(centre[1] - r, centre[3] - r, centre[1] + r, centre[3] + r)) or {}
    local result = {}
    for _, entity in ipairs(found) do
        local mass = entity.MaxMassReclaim or 0
        local energy = entity.MaxEnergyReclaim or 0
        -- q.energy: also props that hold only energy (trees), which AIs like M28 reclaim too
        if (mass > 0 or (q.energy and energy > 0)) and not entity.UnitId then
            local x, y, z = entity:GetPositionXYZ()
            local left = entity.ReclaimLeft or 1
            TableInsert(result, {
                id = entity.EntityId or entity:GetEntityId(), x = x, z = z, mass = mass * left, energy = energy * left,
            })
        end
    end
    if q.limit and TableGetn(result) > q.limit then       -- the most valuable ones (a forest is hundreds)
        table.sort(result, function(a, b) return a.mass + a.energy / 10 > b.mass + b.energy / 10 end)
        for i = TableGetn(result), q.limit + 1, -1 do
            table.remove(result, i)
        end
    end
    return result
end

-- Development aid, off unless the agent sends {"config": {"allowEval": true}}: runs a Lua chunk in the
-- sim and returns its first result (e.g. {"type": "eval", "code": "return GetGameTick()"}). It can do
-- anything the sim can -- spawn units, change resources -- so keep it out of evaluation runs.
Queries.eval = function(q)
    if not Bridge.allowEval then
        error('eval is disabled; enable it with {"config": {"allowEval": true}}', 0)
    end
    local load = rawget(_G, 'AgentBridge_LoadChunk')  -- provided by the engine; the sandbox has no loadstring
    local chunk, message = load(q.code or '', '=agent eval')
    if not chunk then
        error(message, 0)
    end
    return chunk()
end

---------------------------------------------------------------------------------------------------
-- Order capture: another AI plays the observed army (e.g. M28 in the player's slot, /agentplayerai) and
-- every order it gives that army's units through the sim's Issue* functions is recorded, in the bridge's
-- own action format, and handed to the agent with the next observation (obs.orders) -- the AI's play as
-- teacher data. Off unless the agent sends {"config": {"capture": true}}; the bridge's own actions are
-- never recorded. The globals are wrapped when capture is first switched on; AIs that call them at call
-- time (M28's IssueTracked* helpers do) are captured.

local Capture = { on = false, busy = false, shadow = false, wrapped = false, orders = {} }

-- name -> action type, then what each argument after the unit list is
local CaptureSpecs = {
    IssueMove = { 'move', 'pos' }, IssueAggressiveMove = { 'attack_move', 'pos' }, IssuePatrol = { 'patrol', 'pos' },
    IssueAttack = { 'attack', 'target' }, IssueGuard = { 'guard', 'target' }, IssueRepair = { 'repair', 'target' },
    IssueReclaim = { 'reclaim', 'target' }, IssueCapture = { 'capture', 'target' },
    IssueOverCharge = { 'overcharge', 'target' }, IssueTactical = { 'tactical', 'target' }, IssueNuke = { 'nuke', 'pos' },
    IssueTransportLoad = { 'transport_load', 'target' }, IssueTransportUnload = { 'transport_unload', 'pos' },
    IssueFactoryRallyPoint = { 'rally', 'pos' }, IssueBuildMobile = { 'build', 'pos', 'bp' },
    IssueBuildFactory = { 'factory_build', 'bp', 'count' }, IssueUpgrade = { 'upgrade', 'bp' },
    IssueStop = { 'stop' }, IssueClearCommands = { 'clear' }, IssueDive = { 'dive' },
    IssueSiloBuildTactical = { 'silo_build' }, IssueSiloBuildNuke = { 'silo_build', nuke = true },
    IssueScript = { 'enhance', 'script' },
}

-- Records an order another AI gives our army. Returns whether the order was for our army (also when it is
-- one we cannot record).
local function CaptureRecord(spec, args)
    local units = args[1]
    if type(units) ~= 'table' then
        return false
    end
    if units.EntityId then          -- a single unit instead of a list
        units = { units }
    end
    local ids, army = {}, nil
    for _, unit in units do
        if type(unit) == 'table' and not unit.Dead and unit.GetArmy and Contains(Bridge.armies, unit:GetArmy()) then
            army = army or unit:GetArmy()          -- an AI orders its own army's units: one army per order
            if unit:GetArmy() == army then
                TableInsert(ids, unit.EntityId)
            end
        end
    end
    if TableGetn(ids) == 0 then
        return false
    end
    local action = { type = spec[1], units = ids, tick = GetGameTick(), army = army }
    if spec.nuke then
        action.nuke = true
    end
    for k = 2, TableGetn(spec) do
        local what, value = spec[k], args[k]
        if what == 'pos' then
            if type(value) ~= 'table' or value[1] == nil then
                return true
            end
            action.pos = { value[1], value[3] }
        elseif what == 'target' then
            if type(value) == 'table' and value.EntityId then
                action.target = value.EntityId
                -- where it is: props (reclaim) are rarely among the observed ones, and reclaim is an area command
                local ok, p = pcall(value.GetPosition, value)
                if ok and type(p) == 'table' and p[1] then
                    action.target_pos = { p[1], p[3] }
                end
            elseif type(value) == 'table' and value[1] ~= nil and spec[1] == 'attack' then
                action.type = 'attack_ground'     -- a position: fire at the ground, not a command we mirror
                action.pos = { value[1], value[3] }
            else
                return true
            end
        elseif what == 'bp' then
            if type(value) == 'table' then
                value = value.BlueprintId
            end
            if type(value) ~= 'string' then
                return true
            end
            action.bp = value
        elseif what == 'count' then
            action.count = tonumber(value) or 1
        elseif what == 'script' then
            if type(value) ~= 'table' or value.TaskName ~= 'EnhanceTask' or not value.Enhancement then
                return true
            end
            action.name = value.Enhancement
        end
    end
    TableInsert(Capture.orders, action)
    return true
end

local function CaptureWrap()
    if Capture.wrapped then
        return
    end
    Capture.wrapped = true
    for name, spec in CaptureSpecs do
        local original = rawget(_G, name)
        local kind = type(original)
        if kind == 'function' or kind == 'cfunction' then     -- engine functions are 'cfunction' here
            local s = spec
            local f = original
            _G[name] = function(...)
                if Capture.on and not Capture.busy then
                    local ok, ours = pcall(CaptureRecord, s, arg)
                    -- shadow: the other AI's orders for our army are recorded, not given -- what it would do in
                    -- the state the agent has brought the game to (labels for the agent's own games)
                    if Capture.shadow and (ours or not ok) then
                        return nil
                    end
                end
                return f(unpack(arg))
            end
        end
    end
end

-- M28 pauses and unpauses units itself (unit:SetPaused under a mass or energy stall): not an order, so the
-- wrappers above do not see it. In shadow mode it would still act on the agent's army -- the agent's games
-- would not be the agent's own. Its pause functions are looked up in its module at every call.
local function ShadowM28Pauses()
    if Capture.m28 then
        return
    end
    Capture.m28 = true
    local ok, mod = pcall(import, '/mods/M28AI/lua/AI/M28UnitInfo.lua')
    if not ok or type(mod) ~= 'table' then
        return
    end
    for _, name in { 'ForkedPauseUnit', 'PauseOrUnpauseMassUsage', 'PauseOrUnpauseEnergyUsage',
                     'PauseOrUnpauseUnitWithoutTracking' } do
        local f = rawget(mod, name)
        if type(f) == 'function' then
            mod[name] = function(oUnit, a, b, c, d)
                if Capture.shadow and type(oUnit) == 'table' and not oUnit.Dead and oUnit.GetArmy
                        and Contains(Bridge.armies, oUnit:GetArmy()) then
                    return nil
                end
                return f(oUnit, a, b, c, d)
            end
        end
    end
end

-- The orders recorded since the last call (and a fresh list).
local function CaptureTake()
    local orders = Capture.orders
    Capture.orders = {}
    return orders
end
Bridge.capture, Bridge.takeOrders = Capture, CaptureTake     -- for AgentBridge_Observe, defined above

local function Configure(config)
    if type(config.armies) == 'table' and TableGetn(config.armies) > 0 then
        Bridge.armies = config.armies
    end
    if type(config.views) == 'table' then
        Bridge.views = config.views
    end
    if config.capture ~= nil then
        Capture.on = config.capture and true or false
        if Capture.on then
            CaptureWrap()
        end
    end
    if config.shadow ~= nil then          -- may change during a game (the other AI takes over, or hands back)
        Capture.shadow = config.shadow and true or false
        if Capture.shadow then
            ShadowM28Pauses()
        end
    end
    if config.allowEval ~= nil then
        Bridge.allowEval = config.allowEval and true or false
    end
    if config.omniscient ~= nil then
        Bridge.omniscient = config.omniscient and true or false
    end
    if type(config.heightmapSize) == 'number' then
        Bridge.heightmapSize = math.max(4, math.min(512, config.heightmapSize))
    end
    if config.privileged ~= nil then
        Bridge.privileged = config.privileged and true or false
    end
    if config.ghosts ~= nil then
        Bridge.ghosts = config.ghosts and true or false
    end
    if type(config.profile) == 'number' then
        Bridge.profile = config.profile
    end
    if config.nativeObs ~= nil then
        Bridge.nativeObs = config.nativeObs and true or false
    end
    if config.obsCheck ~= nil then
        Bridge.obsCheck = config.obsCheck and true or false
    end
end

local function RunAll(list, handlers)
    local results = {}
    for i, item in ipairs(list or {}) do
        local handler = type(item) == 'table' and handlers[item.type]
        if not handler then
            results[i] = { ok = false, error = 'unknown type ' .. tostring(type(item) == 'table' and item.type or item) }
        else
            local ok, value = pcall(handler, item)
            if ok then
                results[i] = { ok = true, value = value }
            else
                results[i] = { ok = false, error = tostring(value) }
            end
        end
    end
    return results
end

function AgentBridge_Act(msg)
    if type(msg) ~= 'table' then
        return { error = 'expected a JSON object' }
    end
    if type(msg.config) == 'table' then
        Configure(msg.config)
    end
    Capture.busy = true             -- the agent's own orders are not captured
    local ok, result = pcall(function()
        return { actions = RunAll(msg.actions, Actions), queries = RunAll(msg.queries, Queries) }
    end)
    Capture.busy = false
    if not ok then
        error(result, 0)
    end
    return result
end
