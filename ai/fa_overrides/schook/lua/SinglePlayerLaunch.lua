-- faf-neural-ai: hook appended to FAF's /lua/SinglePlayerLaunch.lua (the engine runs /schook
-- files after the file they shadow, in the same module environment). Lets a command-line skirmish
-- (/map ...) pin its AI opponents, which FAF otherwise picks at random on the UI side:
--
--   /agentaifaction <1-4>   faction of every AI slot (1 UEF, 2 Aeon, 3 Cybran, 4 Seraphim)
--   /agentai <key>          personality of every AI slot, a key from /lua/ui/lobby/aitypes.lua
--                           ('rush' is what FAF's command-line launch uses by default)
--   /agentallai             make the player's own slot an AI too (an all-AI game, watched as observer)
--   /agentcheat <x>         cheat and build-rate multiplier of the cheating AIs (AIx: adaptivecheat, rushcheat,
--                           turtlecheat, techcheat, randomcheat), e.g. 1.2; FAF's lobby default is 1.5
--   /agentmods <a,b>        enable these sim mods (by name, e.g. M28AI) for the game; the init file mounts them
--   /agentplayerai <key>    the player's own slot is this AI (e.g. m28ai), still observed by the agent bridge:
--                           another AI's play recorded as teacher data (the bridge's order capture)
--   /agentslots <m,t>       a 1v1 on any map: the player starts at army slot m, one AI at slot t, the other slots
--                           stay empty (the command-line skirmish otherwise puts an AI in every slot)
--
-- With /synclog it also keeps every per-beat sim log instead of only the last few (desync analysis).
--
-- Mounted only by the init files this project generates; normal games never see it.

do
    if HasCommandLineArg("/synclog") then
        ConExecute("sim_KeepAllLogFiles true")
    end

    local factionArg = GetCommandLineArg("/agentaifaction", 1)
    local pinnedFaction = factionArg and tonumber(factionArg[1])
    if pinnedFaction then
        -- SetupCommandLineSkirmish looks this global up at call time. The player's own faction
        -- comes from /faction, so only the AI slots are affected.
        GetRandomFaction = function()
            return pinnedFaction
        end
    end

    local aiArg = GetCommandLineArg("/agentai", 1)
    local pinnedAi = aiArg and aiArg[1]
    local playerAiArg = GetCommandLineArg("/agentplayerai", 1)
    local playerAi = playerAiArg and playerAiArg[1]
    -- GetCommandLineArg returns false (not nil) when absent. Testing ~= nil
    -- silently enabled an AI in the neural player's slot in every game.
    local allAi = HasCommandLineArg("/agentallai") or (playerAi and true or false)
    local cheatArg = GetCommandLineArg("/agentcheat", 1)
    local cheat = cheatArg and tonumber(cheatArg[1])
    local modsArg = GetCommandLineArg("/agentmods", 1)
    local slotsArg = GetCommandLineArg("/agentslots", 1)
    if pinnedAi or allAi or cheat or modsArg or slotsArg then
        local launch = LaunchSinglePlayerSession
        LaunchSinglePlayerSession = function(sessionInfo)
            if slotsArg then
                local slots = {}
                for n in string.gfind(slotsArg[1], '%d+') do
                    table.insert(slots, tonumber(n))
                end
                local armies = sessionInfo.scenarioInfo.Configurations.standard.teams[1].armies
                local mine, theirs = slots[1], slots[2]
                if mine and theirs and armies[mine] and armies[theirs] and mine ~= theirs then
                    local human, ai, rest = nil, nil, {}
                    for _, player in sessionInfo.teamInfo do
                        if player.Human then
                            human = player
                        elseif player.Civilian then
                            table.insert(rest, player)
                        elseif not ai then
                            ai = player
                        end
                    end
                    local team = {}
                    if human then
                        human.ArmyName = armies[mine]
                        human.PlayerColor, human.ArmyColor = 1, 1
                        table.insert(team, human)
                    end
                    if ai then
                        ai.ArmyName = armies[theirs]
                        ai.PlayerColor, ai.ArmyColor = 2, 2
                        table.insert(team, ai)
                    end
                    for _, player in rest do
                        table.insert(team, player)
                    end
                    sessionInfo.teamInfo = team
                end
            end
            if modsArg then
                -- the command-line skirmish only gets the mods selected in the (private, empty) preferences
                local wanted = {}
                for name in string.gfind(modsArg[1], '[^,]+') do
                    wanted[name] = true
                end
                local Mods = import('/lua/mods.lua')
                local uids = {}
                for _, mod in sessionInfo.scenarioMods or {} do
                    uids[mod.uid] = true
                end
                local newest = {}         -- one per name: the newest version
                for uid, mod in Mods.AllMods() do
                    if wanted[mod.name] and (not newest[mod.name] or
                            (tonumber(mod.version) or 0) > (tonumber(newest[mod.name].version) or 0)) then
                        newest[mod.name] = mod
                    end
                end
                for _, mod in newest do
                    uids[mod.uid] = true
                end
                sessionInfo.scenarioMods = Mods.GetGameMods(uids)
                -- what FAF's lobby would add: the tables it always creates, and the defaults of the AI options
                -- the mods declare (lobby.lua ImportModAIOptions); M28 stops in its setup without them
                local options = sessionInfo.scenarioInfo and sessionInfo.scenarioInfo.Options
                if options then
                    options.Ratings = options.Ratings or {}
                    options.ClanTags = options.ClanTags or {}
                    for _, mod in sessionInfo.scenarioMods do
                        local file = mod.location .. '/lua/AI/LobbyOptions/lobbyoptions.lua'
                        local ok, opts = pcall(function() return import(file).AIOpts end)
                        for _, opt in (ok and opts) or {} do
                            local value = opt.values and opt.values[opt.default]
                            if options[opt.key] == nil and value ~= nil then
                                options[opt.key] = type(value) == 'table' and value.key or value
                            end
                        end
                    end
                end
            end
            if cheat then
                -- read by /lua/AI/aiutilities.lua when it buffs a cheating AI (as the lobby's AIx options)
                local options = sessionInfo.scenarioInfo and sessionInfo.scenarioInfo.Options
                if options then
                    options.CheatMult = tostring(cheat)
                    options.BuildMult = tostring(cheat)
                end
            end
            for _, player in sessionInfo.teamInfo or {} do
                if not player.Civilian then
                    if player.Human and allAi then
                        player.Human = false
                        player.AIPersonality = playerAi or pinnedAi or 'rush'
                        player.PlayerName = GetRandomName(player.Faction, player.AIPersonality)
                    elseif not player.Human and pinnedAi and player.AIPersonality then
                        player.AIPersonality = pinnedAi
                        player.PlayerName = GetRandomName(player.Faction, pinnedAi)
                    end
                end
            end
            return launch(sessionInfo)
        end
    end
end
