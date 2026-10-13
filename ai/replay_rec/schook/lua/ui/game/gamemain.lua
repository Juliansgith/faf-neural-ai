-- replay_rec: play the replay as fast as the sim can go (the replay controls start it at normal speed).
local _OnFirstUpdate = OnFirstUpdate
function OnFirstUpdate()
    _OnFirstUpdate()
    ConExecute('WLD_GameSpeed 50')
    ForkThread(function()
        WaitSeconds(2)
        ConExecute('WLD_GameSpeed 50')
    end)
end
