@echo off
rem A game lane on this PC (2026-10-12, the user's go: games on both machines until they say stop): fa_train.player
rem with home _runs\local_player (config.json there: 1 lane of 10 games, the network on the CPU so the GPU keeps a
rem single training process, the same opponents and maps as the server). rl_loop --local-home hands it every new
rem network and collects its finished rounds (lane10_...). Stop: create _runs\local_player\STOP (lanes finish their
rem round first). Closing this window stops the lane at once.
set FAF=C:\faf
cd /d "%FAF%\faf-neural-ai\ai\python"
"%FAF%\ai-venv\Scripts\python.exe" -u -m fa_train.player --home "%FAF%\_runs\local_player" --max-lanes 1 --lane-offset 10 >> "%FAF%\_runs\local_player\player.log" 2>&1
