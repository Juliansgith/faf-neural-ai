# rl_089

Size "M" units-decoder ToolsNet, ~9.2 M weights. Load it with `fa_train.learn_rl.load_net`, or play with
`examples/tools_eval.py --checkpoint models/rl_089.pt`. Model options: `verbs_after_attention`, `fine_positions`.

**How it was made:**
1. **Imitation** of M28's decisions on ~7,400 recorded games (M28 shadowing our army).
2. **Reinforcement learning**, 89 iterations: PPO with V-trace on ~240 recent games each, against
   * FAF's scripted AIs;
   * M28 at a curriculum strength (0.5 → 0.65 cheat);
   * a stronger M28;
   * full M28.

   About 21,000 games in all.
3. **The fine-position head** (where inside a map cell an order goes, ~160 m on any map size) was fitted to M28's
   exact positions on 80 games of 20 km maps, with the rest of the network frozen.

**Results** on 1v1, 5–10 km maps, sampling at temperature 0.5:

| Opponent | Result |
| --- | --- |
| Scripted AIs | 89 % of 240 games |
| M28 0.55 | 76 % of 51 |
| M28 0.6 | 50 % of 114 |
| M28 0.75 | 20 % of 55 |
| Full M28 | 1 of 80 |

On 20 km maps it mostly loses: it tends to walk its commander toward the enemy base.

The checkpoint holds the weights and model options only: training history and paths are removed.
