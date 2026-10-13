-- replay_rec: the sim's records (Sync.ReplayRec, from SimInit.lua's hook) are kept in the preferences file, the one
-- file the user Lua can write: a ring of RING slots, rewritten with every sync that brings records. The recorder
-- process (fa_train.replay_rec) reads the file as it changes and collects every sequence number it has not seen.
local RING = 64
local RecCount = 0
local _OnSync = OnSync
function OnSync()
    local records = Sync.ReplayRec
    if records then
        for _, line in records do
            RecCount = RecCount + 1
            SetPreference('replayrec_' .. math.mod(RecCount, RING), line)
        end
        SavePreferences()
    end
    _OnSync()
end
