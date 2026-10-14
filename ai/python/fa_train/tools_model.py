"""The network for the full-toolbox action space (docs/full-toolbox-design.md, section 2; the first tools
version: docs/tools-design.md, section 4).

tokens ─► MLP ─► transformer ─────────────────┬──► pointer / per-unit selection heads
map planes + tokens scattered into cells ─► ResNet ──► cell head
numbers ─► MLP ─┐
pooled tokens + map summary ─► LSTM (memory) ─► command decoder (MAX_COMMANDS slots) ─► value, belief
blueprint candidates ─► MLP ─► blueprint pointer;  enhancement candidates ─► MLP ─► enhancement pointer

Each slot chooses a verb, then its subject (one unit, or a set), then the verb's arguments in a fixed order:
blueprint (build, produce: a pointer over everything the chosen builder can build now), enhancement, option
(fire state, missile kind, ability bit), count, placement (a spot in a map cell, or against one of our
buildings for the adjacency bonus), map cell, spot, target unit. Every choice feeds the next through `ctx`.

One class, three sizes (SIZES). `ToolsNet.forward_step` acts (sampling or greedy); `ToolsNet.evaluate` scores
given actions (teacher forcing) for cloning and PPO. A `temperature` below 1 sharpens every head: an order stays in
force until it is replaced, so a random order every few decisions undoes everything (night 1), and training games
act -- and PPO scores them -- at a low temperature. Actions use token and candidate indices; `commands_to_indices`
and `indices_to_commands` translate to and from fa_env.tools commands (unit ids, blueprint ids, names).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from fa_env.tools import (CELL_VERBS, COUNTS, FINE, FINE_UNITS, FIRE_STATES, GRID, MAX_COMMANDS, N_OPTIONS, OPTION_VERBS, PLACES,
                          SET_SUBJECT, SILO_KINDS, SINGLE_SUBJECT, SPOTS, TARGET_ROWS, TARGET_VERBS, VERBS)
from fa_env.tools_obs import (CAND_KEYS, CAND_MOBILE, ENH_KEYS, MAP_CHANNELS, N_CAND_FEATURES, N_ENH_FEATURES,
                              N_FOCUS,
                              N_FEATURES, N_LAST_FEATURES, N_ORDERS, N_SCALARS, PAIR_KEYS, PRIVILEGED, STEP_KEYS,
                              TOKEN_KEYS,
                              TOKEN_T_KEYS)

V, NC, G2 = len(VERBS), len(COUNTS), GRID * GRID
NP, NSP, NR = len(PLACES), SPOTS * SPOTS, len(TARGET_ROWS)
K = MAX_COMMANDS
VOCAB = 1024
ENH_VOCAB = 256
SIZES = {
    "S": dict(d=64, layers=2, heads=4, map_ch=32, core=256),
    "M": dict(d=128, layers=3, heads=4, map_ch=64, core=512),
    "L": dict(d=256, layers=4, heads=8, map_ch=96, core=1024),
}
V_END, V_BUILD, V_PRODUCE, V_UPGRADE, V_ENHANCE = (VERBS.index(v) for v in ("end", "build", "produce", "upgrade",
                                                                             "enhance"))
V_FIRE, V_SILO, V_TOGGLE = (VERBS.index(v) for v in ("fire_state", "silo_build", "toggle"))
V_CMOVE, V_CAMOVE = (VERBS.index(v) for v in ("commander_move", "commander_attack_move"))
NEG = -1e9
BUILD_BUDGET = 3


def _verb_table(values) -> torch.Tensor:
    return torch.tensor(values)


IS_SINGLE = _verb_table([v in SINGLE_SUBJECT for v in VERBS])
IS_SET = _verb_table([v in SET_SUBJECT for v in VERBS])
NEEDS_CELL = _verb_table([v in CELL_VERBS for v in VERBS])
HAS_OPTION = _verb_table([v in OPTION_VERBS for v in VERBS])
TARGET_ROW = _verb_table([TARGET_ROWS.index(TARGET_VERBS[v]) if v in TARGET_VERBS else -1 for v in VERBS])
FIRE_OPTIONS = _verb_table([k < len(FIRE_STATES) for k in range(N_OPTIONS)])
ACTION_KEYS = ("verb", "single", "set", "bp", "enh", "option", "count", "place", "anchor", "cell", "spot", "target")
ENTROPY_KEYS = ("verb", "subject", "blueprint", "enhancement", "option", "count", "place", "cell", "target", "sub")
# the 'units' decoder: one choice per own unit (0 = nothing, else a verb) with that unit's arguments, [T] each
# usub (2026-10-12): the fine pixel of the cell, fz * FINE + fx (maps above 5 km). Last: recorded actions are
# positional lists (tools_trainer.compact_action), older ones simply end before it
UNIT_KEYS = ("uverb", "ubp", "uenh", "uoption", "ucount", "uplace", "uanchor", "ucell", "uspot", "utarget", "usub")
UNIT_METRIC_KEYS = ("unit_none", "unit_ordered", "unit_precision", "unit_verb", "blueprint", "enhancement", "option",
                    "count", "place", "anchor", "cell", "cell_near", "spot", "target", "sub", "sub_near")
METRIC_KEYS = ("verb", "subject", "set_unit", "set_leader", "set_recall", "set_precision", "set_nonempty", "blueprint", "enhancement", "option", "count", "place", "anchor", "cell",
               "cell_near", "spot", "target")


class SDPALayer(nn.TransformerEncoderLayer):
    """nn.TransformerEncoderLayer (norm_first, same parameters and state dict) whose attention runs through
    scaled_dot_product_attention. In training the stock layer keeps the [batch, heads, T, T] attention weights for
    the backward pass: 640 decisions of 500 tokens are 2.5 GB per layer in bf16, which is what filled the 24 GB
    (2026-10-06). The fused kernels recompute them instead. FA_TOOLS_MATH_ATTENTION=1 restores the stock path."""

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if os.environ.get("FA_TOOLS_MATH_ATTENTION") == "1":
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)
        x = src
        h = self.norm1(x)
        B, T, d = h.shape
        heads = self.self_attn.num_heads
        qkv = F.linear(h, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.view(B, T, 3, heads, d // heads).permute(2, 0, 3, 1, 4)              # 3 x [B,heads,T,dh]
        mask = None
        if src_key_padding_mask is not None:
            kpm = src_key_padding_mask
            if kpm.dtype != torch.bool:              # nn.TransformerEncoder hands the layers a float mask (-inf)
                kpm = kpm != 0
            allowed = ~kpm                                                                   # [B,T]
            allowed = allowed | ~allowed.any(-1, keepdim=True)        # a row with nothing to attend to: everything
            mask = allowed.view(B, 1, 1, T)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if getattr(self, "capture_attn", False):       # the watch debugger: the weights, averaged over heads
            with torch.no_grad():
                w = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
                if mask is not None:
                    w = w.masked_fill(~mask, float("-inf"))
                self.last_attn = torch.softmax(w, -1).mean(1)                              # [B,T,T]
        x = x + self.self_attn.out_proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.linear2(self.activation(self.linear1(self.norm2(x))))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


def _safe(mask: torch.Tensor) -> torch.Tensor:
    """Rows with nothing allowed get their first option allowed (they are never scored), so softmaxes stay finite."""
    empty = ~mask.any(-1, keepdim=True)
    first = torch.zeros_like(mask)
    first[..., 0] = True
    return mask | (empty & first)


def _pick(logits: torch.Tensor, label: torch.Tensor | None, mode: str, top_p: float | None = None) -> torch.Tensor:
    if label is not None:
        return label.clamp(min=0)
    if mode == "greedy":
        return logits.argmax(-1)
    if top_p is not None:                  # nucleus: sample among the most likely choices holding `top_p` of the mass
        ordered, order = logits.sort(-1, descending=True)
        before = ordered.softmax(-1).cumsum(-1) - ordered.softmax(-1)
        ordered = ordered.masked_fill(before >= top_p, NEG)
        logits = torch.full_like(logits, NEG).scatter(-1, order, ordered)
    u = torch.rand_like(logits).clamp_(1e-10, 1.0)            # Gumbel-max: a categorical sample in 4 kernels
    return (logits - torch.log(-torch.log(u))).argmax(-1)


def _cat_terms(logits, choice, smoothing, mask, full: bool = True):
    """(log p(choice), entropy, smoothed cross-entropy) of a masked categorical; `full=False` (acting) computes
    only the log-probability."""
    logp_all = F.log_softmax(logits, -1)
    logp = logp_all.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    if not full:
        return logp, 0.0, 0.0
    p = logp_all.exp()
    ent = -(p * logp_all.masked_fill(~mask, 0.0)).sum(-1)
    if smoothing > 0:
        n = mask.sum(-1).clamp(min=1)
        uniform = (logp_all.masked_fill(~mask, 0.0)).sum(-1) / n
        ce = -((1 - smoothing) * logp + smoothing * uniform)
    else:
        ce = -logp
    return logp, ent, ce


def _reduce_set_bc(per_unit, targets, eligible, balanced=False):
    """Mean BCE, optionally giving each present label class equal weight per command.

    Balancing only changes the cloning objective; sampling and PPO likelihoods
    remain the same independent Bernoulli distribution.
    """
    if not balanced:
        return per_unit.masked_fill(~eligible, 0.0).sum(-1) / eligible.sum(-1).clamp_min(1)
    positive, negative = targets & eligible, ~targets & eligible
    npos, nneg = positive.sum(-1), negative.sum(-1)
    pos = per_unit.masked_fill(~positive, 0.0).sum(-1) / npos.clamp_min(1)
    neg = per_unit.masked_fill(~negative, 0.0).sum(-1) / nneg.clamp_min(1)
    classes = (npos > 0).float() + (nneg > 0).float()
    return (pos + neg) / classes.clamp_min(1)


class ToolsNet(nn.Module):
    def __init__(self, size: str = "M", amp: bool = False, decoder: str = "slots"):
        super().__init__()
        if decoder not in ("slots", "units"):
            raise ValueError(f"unknown decoder {decoder!r}")
        # 'slots': up to MAX_COMMANDS commands one after another, each pointing at its units (the first design).
        # 'units': every own unit decides for itself, all at once -- nothing, or a verb with its arguments. Cloning
        # M28 with slots repeated verbs until the slots were full and starved production and the army
        # (2026-10-05); per unit there is no list to run on, and no limit on how many units get orders.
        self.decoder = decoder
        cfg = SIZES[size]
        self.size = size
        # bf16 for the encoder (transformer and map network, where the tensor cores pay off); the memory, the
        # decoder and every probability stay in float32, which PPO's ratios need. Acting and training must
        # agree on it, or the same action gets two different log-probabilities.
        self.amp = amp
        self.bc_set_loss_mode = "mean"
        # builds per decision when the policy acts (and PPO scores its own actions); None while cloning. M28 gives
        # 1-1.5 per 2 s; the clone, decoding one command after another, kept adding builds -- ~15 a decision, its 24
        # slots full and production and army orders never reached (2026-10-05)
        self.build_budget: int | None = BUILD_BUDGET
        # research switch for playing a clone (not for PPO: probabilities are not corrected for it): sample every
        # choice from its most likely options holding this much probability. Greedy play turns a teacher's mix
        # (M28's early factories: 57 % engineers, 21 % tanks, 19 % scouts) into its mode (engineers only)
        self.act_top_p: float | None = None
        # units decoder, acting: verbs a unit would get with less than this probability are not chosen. Hundreds of
        # units each sampling a 0.1 % verb gave a stray hold-fire, pause or toggle every few decisions, and those
        # stay in force (the first sampled games, 2026-10-05). Scoring ignores the cut (at most this much mass).
        self.act_min_p: float = 0.02
        # acting: the commander's destination is its most likely cell, not a sample -- one sampled step toward
        # the enemy with a damaged commander loses the game (lost games, 2026-10-06: the policy sent a commander
        # under half health toward the enemy 33 % of the time, M28 9 %)
        self.commander_greedy: bool = False
        # a model option (MODEL_OPTIONS): saved with a checkpoint, applied by every loader (apply_model_options)
        self.verbs_after_attention: bool = False
        # a model option: the fine pixel inside a chosen cell on maps above 5 km (a network trained with the fine
        # head); off, orders go to the cell (centre) as before -- an older network never samples an untrained head
        self.fine_positions: bool = False
        d, core, mc = cfg["d"], cfg["core"], cfg["map_ch"]
        self.d, self.core_dim, self.map_ch = d, core, mc
        # tokens
        self.bp_embed = nn.Embedding(VOCAB, 16)
        self.order_embed = nn.Embedding(N_ORDERS, 8)
        self.last_embed = nn.Embedding(V, 8)                 # the verb of the last order given to the unit
        self.tok_in = nn.Sequential(nn.Linear(N_FEATURES + 24 + 8 + N_LAST_FEATURES, d), nn.ReLU(),
                                    nn.Linear(d, d))
        layer = SDPALayer(d, cfg["heads"], 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        # training: recompute the encoder and the decoder in the backward pass instead of keeping their
        # activations (the whole batch's token embeddings, map features and per-unit logits): roughly half the
        # memory for ~30 % more time. Set by the trainer (tools_bc --grad-checkpoint).
        self.grad_checkpoint: bool = False
        # the watch debugger (tools_eval --watch): forward_step keeps what the decision was made of in last_debug
        self.debug: bool = False
        self.last_debug: dict | None = None
        self.tok_norm = nn.LayerNorm(d)
        # blueprint and enhancement candidates
        self.cand_in = nn.Sequential(nn.Linear(N_CAND_FEATURES + 16, d), nn.ReLU(), nn.Linear(d, d))
        self.enh_embed = nn.Embedding(ENH_VOCAB, 16)
        self.enh_in = nn.Sequential(nn.Linear(N_ENH_FEATURES + 16, d), nn.ReLU(), nn.Linear(d, d))
        # map
        self.scatter = nn.Linear(d, 16)
        self.map_in = nn.Conv2d(len(MAP_CHANNELS) + 16, mc, 3, padding=1)
        self.map_blocks = nn.Sequential(*[ResBlock(mc) for _ in range(3)])
        self.map_down = nn.Sequential(nn.Conv2d(mc, mc, 4, stride=4), nn.ReLU(), nn.Conv2d(mc, mc, 4, stride=4), nn.ReLU())
        self.map_vec = nn.Linear(mc * 4, d)
        # Close-ups (2026-10-12): FINE x FINE windows of FINE_UNITS-unit pixels (~160 m on any map) around our
        # commander, our army and the fight (observation "focus"), seen as the fine head sees a cell -- the map
        # features interpolated, the tokens at their pixels -- pooled into the core's input. The coarse map is
        # 2.5 km a cell on an 81 km map; these are not. The projection starts at zero: an older network plays on.
        self.focus_map = nn.Conv2d(mc, 8, 1)
        self.focus_tok = nn.Linear(d, 8)
        self.focus_net = nn.Sequential(nn.Conv2d(18, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
                                       nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.focus_out = nn.Linear(N_FOCUS * 64, core)
        nn.init.zeros_(self.focus_out.weight)
        nn.init.zeros_(self.focus_out.bias)
        # numbers and core
        self.scal = nn.Sequential(nn.Linear(N_SCALARS, d), nn.ReLU())
        self.core_in = nn.Sequential(nn.Linear(4 * d + d + d, core), nn.ReLU())
        self.lstm = nn.LSTM(core, core, batch_first=True)
        # command decoder
        self.verb_head = nn.Linear(core, V)
        self.verb_embed = nn.Embedding(V, core)
        self.subj_q = nn.Linear(core, d)
        self.set_bias = nn.Linear(core, 1)
        self.lead_proj = nn.Linear(d, core)      # a set's leader, for choosing who joins it
        self.member_q = nn.Linear(core, d)
        self.actor_proj = nn.Linear(d, core)
        self.bp_q = nn.Linear(core, d)
        self.bp_proj = nn.Linear(d, core)
        self.enh_q = nn.Linear(core, d)
        self.enh_proj = nn.Linear(d, core)
        self.option_head = nn.Linear(core, N_OPTIONS)
        self.option_embed = nn.Embedding(N_OPTIONS, core)
        self.count_head = nn.Linear(core, NC)
        self.place_head = nn.Linear(core, NP)
        self.place_embed = nn.Embedding(NP, core)
        self.anchor_q = nn.Linear(core, d)
        self.anchor_proj = nn.Linear(d, core)
        self.spot_head = nn.Linear(core, NSP)
        self.cell_q = nn.Linear(core, mc)
        self.cell_bias = nn.Conv2d(mc, 1, 1)
        self.cell_proj = nn.Linear(mc, core)
        # The fine pixel inside a chosen cell (maps above 5 km): a close-up of FINE x FINE pixels around the cell --
        # the map features interpolated, the tokens at their exact positions, which pixels are the cell's --
        # scored per pixel against the unit's context. Zero-initialised query, bias and feedback: a fresh head
        # chooses uniformly and changes nothing upstream (fine_positions is off until it is trained).
        self.fine_map = nn.Conv2d(mc, 8, 1)
        self.fine_tok = nn.Linear(d, 8)
        self.fine_net = nn.Sequential(nn.Conv2d(18, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.fine_q = nn.Linear(core, 32)
        self.fine_bias = nn.Conv2d(32, 1, 1)
        self.fine_proj = nn.Linear(32, core)
        for layer in (self.fine_q, self.fine_bias, self.fine_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.tgt_q = nn.Linear(core, d)
        self.tgt_proj = nn.Linear(d, core)
        self.slot_cell = nn.GRUCell(core, core)
        if decoder == "units":
            self.unit_in = nn.Sequential(nn.Linear(d + core, core), nn.ReLU(), nn.Linear(core, core))
            # Two passes: every unit guesses its verb alone, then one attention layer over the own units, each
            # carrying its guess, and the verbs and arguments are decided from that. Which of ten idle engineers
            # takes a build is a question between them; the first pass answered it for each alone (2026-10-06).
            # The layer starts as the identity (zero output projections), so a one-pass checkpoint plays on.
            self.unit_attn = SDPALayer(core, 4, 2 * core, dropout=0.0, batch_first=True, norm_first=True)
            nn.init.zeros_(self.unit_attn.self_attn.out_proj.weight)
            nn.init.zeros_(self.unit_attn.self_attn.out_proj.bias)
            nn.init.zeros_(self.unit_attn.linear2.weight)
            nn.init.zeros_(self.unit_attn.linear2.bias)
            self.intent_in = nn.Linear(V, core, bias=False)      # a unit's first-pass verb probabilities, embedded
            nn.init.zeros_(self.intent_in.weight)
        # local critic (fa_train.learn_rl, spatial credit): per map cell, the coming local trade in mass. Not used
        # for acting; checkpoints from before it load with a fresh one (load_state_dict, OPTIONAL_PREFIXES)
        self.local_value = nn.Conv2d(mc, 1, 1)
        nn.init.zeros_(self.local_value.weight)
        nn.init.zeros_(self.local_value.bias)
        # value (privileged inputs only here) and belief
        self.value_head = nn.Sequential(nn.Linear(core + 2 * len(PRIVILEGED), 256), nn.ReLU(), nn.Linear(256, 1))
        # the opponent the game is against (learn_rl --opponent-critic; never seen when playing): one pool mixes AIs
        # we beat 86 % of the time, M28 at three cheat levels (58 % .. 5 %) and our own networks, and a critic
        # blind to which of them it faces turns "a hard opponent" into every decision's advantage. Added to the
        # value, zero at the start (an older checkpoint's values are unchanged), trained by the critic's loss only
        self.opp_value = nn.Sequential(nn.Linear(core + N_OPPONENT, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.opp_value[2].weight)
        nn.init.zeros_(self.opp_value[2].bias)
        self.belief_head = nn.Sequential(nn.Linear(core, 256), nn.ReLU(), nn.Linear(256, len(PRIVILEGED)))
        for head in (self.verb_head, self.option_head, self.count_head, self.place_head, self.spot_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    OPTIONAL_PREFIXES = ("local_value.", "fine_", "focus_", "opp_")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """As nn.Module's, except that parameters added later (OPTIONAL_PREFIXES) may be missing from older
        checkpoints: they keep this network's fresh values."""
        own = super().state_dict()
        missing = {k: v for k, v in own.items() if k.startswith(self.OPTIONAL_PREFIXES) and k not in state_dict}
        if missing:
            state_dict = {**state_dict, **missing}
        # map channels added later (the danger maps, 2026-10-11) come in after the old map channels and before
        # the 16 grid channels: an older checkpoint's map_in gets zero weights for them (it acts as before)
        w = state_dict.get("map_in.weight")
        new_w = own["map_in.weight"]
        if w is not None and w.shape[1] < new_w.shape[1]:
            old_maps = w.shape[1] - 16
            grown = torch.zeros(w.shape[0], new_w.shape[1], *w.shape[2:], dtype=w.dtype, device=w.device)
            grown[:, :old_maps] = w[:, :old_maps]
            grown[:, -16:] = w[:, old_maps:]
            state_dict = {**state_dict, "map_in.weight": grown}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------ encoder
    def encode(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-observation encoding: token embeddings [B,T,d], map features [B,C,G,G], core input [B,core]."""
        tok, tmask = b["tokens"], b["token_mask"]
        B, T, _ = tok.shape
        x = torch.cat([tok, self.bp_embed(b["token_bp"].clamp(0, VOCAB - 1)),
                       self.order_embed(b["token_order"].clamp(0, N_ORDERS - 1)),
                       self.last_embed(b["token_last"].clamp(0, V - 1)), b["token_last_feat"]], -1)
        h = self.tok_in(x)
        pad = ~tmask & tmask.any(1, keepdim=True)          # an observation without tokens attends to padding
        h = self.transformer(h, src_key_padding_mask=pad)
        h = self.tok_norm(h) * tmask.unsqueeze(-1)
        # tokens scattered into their map cells
        cx = (tok[..., 7] * GRID).long().clamp(0, GRID - 1)
        cz = (tok[..., 8] * GRID).long().clamp(0, GRID - 1)
        idx = (cz * GRID + cx).unsqueeze(-1).expand(-1, -1, 16)
        sc = self.scatter(h) * tmask.unsqueeze(-1)
        grid = torch.zeros(B, G2, 16, device=tok.device, dtype=sc.dtype).scatter_add_(1, idx, sc)
        grid = grid.transpose(1, 2).reshape(B, 16, GRID, GRID)
        m = self.map_blocks(self.map_in(torch.cat([b["map"], grid], 1)))
        map_vec = F.relu(self.map_vec(self.map_down(m).flatten(1)))
        # token summaries: mean of own, mean of enemy-side (seen, remembered, radar), mean of the rest, max of all
        ttype = b["token_type"]
        pooled = []
        for sel in ((ttype == 0), (ttype >= 1) & (ttype <= 3), (ttype >= 4)):
            w = (sel & tmask).unsqueeze(-1).to(h.dtype)
            pooled.append((h * w).sum(1) / w.sum(1).clamp(min=1.0))
        pooled.append(h.masked_fill(~tmask.unsqueeze(-1), -1e4).max(1).values.clamp(min=-10.0))
        core_in = self.core_in(torch.cat(pooled + [map_vec, self.scal(b["scalars"])], -1))
        if "focus" in b and "fine" in b:
            core_in = core_in + self.focus_out(self._close_ups(h, m, b)).to(core_in.dtype)
        return h, m, core_in

    def _close_ups(self, h, m, b) -> torch.Tensor:
        """[B, N_FOCUS * 64]: each close-up's pixels (FINE x FINE, FINE_UNITS game units each, centred on its
        focus point), mean- and max-pooled; zero where the point is absent."""
        foc = b["focus"].float()                                                                  # [B,K,3]
        B, K, _ = foc.shape
        dev = h.device
        size = b["fine"][:, 2:4].float().clamp(min=1.0)                                           # [B,2] W, H
        span = (FINE * FINE_UNITS) / size                                                         # window, map fractions
        k = (torch.arange(FINE, device=dev, dtype=torch.float32) + 0.5) / FINE - 0.5
        px = foc[..., 0:1] + k * span[:, 0].view(B, 1, 1)                                        # [B,K,F]
        pz = foc[..., 1:2] + k * span[:, 1].view(B, 1, 1)
        grid = torch.stack([px.unsqueeze(2).expand(-1, -1, FINE, -1), pz.unsqueeze(3).expand(-1, -1, -1, FINE)], -1)
        fm = self.focus_map(m).float()                                                            # [B,8,G,G]
        f0 = F.grid_sample(fm.repeat_interleave(K, 0), grid.view(B * K, FINE, FINE, 2) * 2 - 1, mode="bilinear",
                           padding_mode="zeros", align_corners=False)                             # [BK,8,F,F]
        tok = b["tokens"].float()
        wx = ((tok[..., 7].unsqueeze(1) - foc[..., 0:1]) / span[:, 0].view(B, 1, 1) * FINE + FINE / 2).floor().long()
        wz = ((tok[..., 8].unsqueeze(1) - foc[..., 1:2]) / span[:, 1].view(B, 1, 1) * FINE + FINE / 2).floor().long()
        inside = b["token_mask"].unsqueeze(1) & (wx >= 0) & (wx < FINE) & (wz >= 0) & (wz < FINE)  # [B,K,T]
        widx = (wz.clamp(0, FINE - 1) * FINE + wx.clamp(0, FINE - 1)).view(B * K, -1)
        T = tok.shape[1]
        vals = (self.focus_tok(h).float().unsqueeze(1) * inside.unsqueeze(-1)).view(B * K, T, 8)
        FF = FINE * FINE
        f1 = torch.zeros(B * K, FF, 8, device=dev).scatter_add_(1, widx.unsqueeze(-1).expand(-1, -1, 8), vals)
        cnt = torch.zeros(B * K, FF, device=dev).scatter_add_(1, widx, inside.view(B * K, T).float())
        on = foc[..., 2].reshape(B * K, 1, 1, 1)
        feat = torch.cat([f0, f1.transpose(1, 2).reshape(B * K, 8, FINE, FINE),
                          torch.log1p(cnt).view(B * K, 1, FINE, FINE), on.expand(-1, 1, FINE, FINE)], 1) * on
        P = self.focus_net(feat.to(h.dtype)).flatten(2)                                            # [BK,32,FF]
        pooled = torch.cat([P.mean(-1), P.amax(-1)], -1) * on.view(B * K, 1).to(P.dtype)           # [BK,64]
        return pooled.view(B, K * 64)

    def candidates(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Blueprint candidates [B,C,d] and enhancement candidates [B,E,d]."""
        ch = self.cand_in(torch.cat([b["cand_feat"], self.bp_embed(b["cand_bp"].clamp(0, VOCAB - 1))], -1))
        eh = self.enh_in(torch.cat([b["enh_feat"], self.enh_embed(b["enh_idx"].clamp(0, ENH_VOCAB - 1))], -1))
        return ch * b["cand_mask"].unsqueeze(-1), eh * b["enh_mask"].unsqueeze(-1)

    # ------------------------------------------------------------------ command decoder
    def decode(self, h, m, z, b, actions: dict | None = None, mode: str = "sample", smoothing: float = 0.0,
               temperature: float = 1.0):
        """Choose (or, with `actions`, score) a decision's commands with this network's decoder."""
        if self.decoder == "units":
            return self.decode_units(h, m, z, b, actions, mode, smoothing, temperature)
        return self.decode_slots(h, m, z, b, actions, mode, smoothing, temperature)

    def decode_units(self, h, m, z, b, actions: dict | None = None, mode: str = "sample", smoothing: float = 0.0,
                     temperature: float = 1.0):
        """Every own unit at once: nothing (0) or a verb, then that unit's arguments, chosen as the slots decoder
        chooses a single-subject command's (the argument heads are the same). With `actions` (UNIT_KEYS, [B,T]
        each; `uverb` -1 = unknown, not scored) it scores them. Returns (actions, logp [B], entropy dict of [B],
        bc loss [B], metrics)."""
        inv_t = 1.0 / temperature
        top_p = self.act_top_p if actions is None else None
        B, T, d = h.shape
        dev = h.device
        full = actions is not None
        mflat = m.flatten(2)                                   # [B,C,G2]
        cell_bias = self.cell_bias(m).flatten(1)               # [B,G2]
        has_option, needs_cell = HAS_OPTION.to(dev), NEEDS_CELL.to(dev)
        target_row, fire_options = TARGET_ROW.to(dev), FIRE_OPTIONS.to(dev)
        cand_h, enh_h = self.candidates(b)                     # [B,C,d], [B,E,d]
        cand_kind = b["cand_kind"]
        tokmask = b["token_mask"]
        scale_d, scale_c = 1.0 / math.sqrt(d), 1.0 / math.sqrt(self.map_ch)
        ent = {k: torch.zeros(B, device=dev) for k in ENTROPY_KEYS}
        metrics = {k: [torch.zeros(B, device=dev), torch.zeros(B, device=dev)] for k in UNIT_METRIC_KEYS}

        # 1. every unit: nothing, or a verb it can be given now
        ctx_all = self.unit_in(torch.cat([h, z.unsqueeze(1).expand(-1, T, -1)], -1))        # [B,T,core]
        vmask = (b["mask_subject"] & tokmask.unsqueeze(1)).transpose(1, 2) & b["mask_verb"].unsqueeze(1)  # [B,T,V]
        vmask[..., V_END] = False
        actor = vmask.any(-1)                                  # [B,T] units that can be given something
        vmask[..., V_END] = True
        # Nothing-or-a-verb is a rate -- how often a unit in this state gets an order -- not a best guess: it is
        # sampled at temperature 1 in every mode. Greedy (and a temperature below 1) never fires what the teacher
        # does less than half the time per decision: the teacher's clone gave 0.06 builds a decision and lost
        # every game to easy (2026-10-05). `temperature` and `mode` apply to the arguments.
        first = self.verb_head(ctx_all).masked_fill(~vmask, NEG)
        intent = self.intent_in(F.softmax(first, -1)) * actor.unsqueeze(-1).float()
        # the own units side by side (the enemy, spots and wrecks are already in h): attention over up to a few
        # hundred rows, not MAX_TOKENS
        n_own = actor.sum(1)
        Uo = max(1, int(n_own.max()))
        own_idx = torch.argsort((~actor).to(torch.uint8), dim=1, stable=True)[:, :Uo]
        own_on = torch.arange(Uo, device=dev).unsqueeze(0) < n_own.unsqueeze(1)                # [B,Uo]
        gather_idx = own_idx.unsqueeze(-1).expand(-1, -1, ctx_all.shape[-1])
        x = (ctx_all + intent).gather(1, gather_idx)
        x = self.unit_attn(x, src_key_padding_mask=~own_on)
        x = torch.where(own_on.unsqueeze(-1), x, ctx_all.gather(1, gather_idx))
        ctx_all = ctx_all.scatter(1, gather_idx, x)
        # The final verbs. Until 2026-10-09 they were the first pass's (`first`, computed before the attention): the
        # second pass reached only the arguments, never the choice it was built for -- which of ten idle engineers
        # takes the build. `verbs_after_attention` (a model option, saved with the checkpoint) decides them from the
        # attended context; the attention starts as the identity, so a network gets it only by training with it.
        vlog = self.verb_head(ctx_all).masked_fill(~vmask, NEG) if self.verbs_after_attention else first
        # an optional per-unit weight of the loss (actions["uweight"], [B,T]): what a unit's lesson counts
        uw = actions.get("uweight") if actions is not None else None
        if actions is not None:
            known = actor & (actions["uverb"] >= 0)
            uverb = actions["uverb"].clamp(min=0).long() * actor
        else:
            known = actor
            pick = vlog
            if self.act_min_p:
                rare = F.softmax(vlog, -1) < self.act_min_p
                rare[..., V_END] = False
                pick = vlog.masked_fill(rare, NEG)
            uverb = _pick(pick, None, "sample", top_p) * actor
        lp_all = F.log_softmax(vlog, -1)
        lp_v = lp_all.gather(-1, uverb.unsqueeze(-1)).squeeze(-1)
        knownf = known.float()
        unit_lp = lp_v * knownf                                # [B,T] each unit's own log-probability (verb + args)
        logp = unit_lp.sum(1)
        loss = torch.zeros(B, device=dev)
        if full:
            pr = lp_all.exp()
            ent["verb"] = ((-(pr * lp_all.masked_fill(~vmask, 0.0)).sum(-1)) * actor.float()).sum(1)
            # No label smoothing here (it stays on the arguments): smoothing asks every unit to give each verb it
            # could be given a little probability, every decision. With a hundred units that was two stray orders
            # a decision -- hold fire 0.9, stop 0.3, pause 0.25 -- which stay in force: paused factories, an army
            # on hold fire (the first units clones, 2026-10-05).
            loss = (-lp_v * knownf * (uw if uw is not None else 1.0)).sum(1)
            guess = vlog.argmax(-1)
            pos, said = known & (uverb != V_END), known & (guess != V_END)
            for name, hit, tot in (("unit_none", known & ~pos & ~said, known & ~pos), ("unit_ordered", pos & said, pos),
                                   ("unit_precision", pos & said, said), ("unit_verb", pos & (guess == uverb), pos)):
                metrics[name][0] += hit.float().sum(1)
                metrics[name][1] += tot.float().sum(1)

        # 2. the units given a verb, side by side as [B,U] (U: the most in any decision of the batch)
        act = actor & (uverb != V_END)
        n_act = act.sum(1)
        U = max(1, int(n_act.max()))
        idx = torch.argsort((~act).to(torch.uint8), dim=1, stable=True)[:, :U]              # their tokens, in order
        on = torch.arange(U, device=dev).unsqueeze(0) < n_act.unsqueeze(1)                  # [B,U]
        verb = uverb.gather(1, idx) * on
        take = lambda x: x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))  # noqa: E731  rows of [B,T,X]
        row_of = lambda keys, i: keys.gather(1, i.unsqueeze(-1).expand(-1, -1, keys.shape[-1]))  # noqa: E731
        ctx = take(ctx_all) + self.verb_embed(verb)
        uw_u = uw.gather(1, idx) if uw is not None else None                                  # [B,U]
        lab = (lambda key: actions[key].gather(1, idx)) if actions is not None else (lambda key: None)
        given = (lambda key, o: o & (actions[key].gather(1, idx) >= 0)) if actions is not None else (lambda key, o: o)
        arg_lp = torch.zeros(B, U, device=dev)                 # the arguments' log-probabilities per acting unit

        def score(logits, choice, mask, o, ent_key, name=None):
            nonlocal logp, loss, arg_lp
            lp, e, ce = _cat_terms(logits, choice, smoothing, mask, full)
            of = o.float()
            logp = logp + (lp * of).sum(1)
            arg_lp = arg_lp + lp * of
            if full:
                ent[ent_key] = ent[ent_key] + (e * of).sum(1)
                loss = loss + (ce * of * (uw_u if uw_u is not None else 1.0)).sum(1)
                if name is not None:
                    metrics[name][0] += ((logits.argmax(-1) == choice).float() * of).sum(1)
                    metrics[name][1] += of.sum(1)

        # blueprint (build: a structure the builder can build now; produce: a unit; upgrade: which, if a choice)
        is_mobile = (cand_kind == CAND_MOBILE).unsqueeze(1)                                 # [B,1,C]
        row = take(b["can_build"] & b["cand_mask"].unsqueeze(1))                            # [B,U,C]
        bld, prd = on & (verb == V_BUILD), on & (verb == V_PRODUCE)
        structures = row & ~is_mobile
        upg = on & (verb == V_UPGRADE) & structures.any(-1)
        bmask = _safe(torch.where((bld | upg).unsqueeze(-1), structures, row & is_mobile))
        b_logits = (torch.einsum("bcd,bud->buc", cand_h, self.bp_q(ctx)) * (scale_d * inv_t)).masked_fill(~bmask, NEG)
        bp = _pick(b_logits, lab("ubp"), mode, top_p)
        bp_on = bld | prd | upg
        score(b_logits, bp, bmask, bp_on, "blueprint", "blueprint")
        ctx = ctx + self.bp_proj(row_of(cand_h, bp)) * bp_on.unsqueeze(-1)
        bkind = cand_kind.gather(1, bp).clamp(0, CAND_MOBILE)
        # enhancement
        enh_on = on & (verb == V_ENHANCE)
        emask = _safe(take(b["can_enh"] & b["enh_mask"].unsqueeze(1)))
        e_logits = (torch.einsum("bed,bud->bue", enh_h, self.enh_q(ctx)) * (scale_d * inv_t)).masked_fill(~emask, NEG)
        enh = _pick(e_logits, lab("uenh"), mode, top_p)
        score(e_logits, enh, emask, enh_on, "enhancement", "enhancement")
        ctx = ctx + self.enh_proj(row_of(enh_h, enh)) * enh_on.unsqueeze(-1)
        # option: fire state, missile kind, or which ability to switch
        opt_on = has_option[verb] & on
        silo = F.pad(b["mask_silo"], (0, N_OPTIONS - len(SILO_KINDS)))
        omask = _safe(torch.where((verb == V_FIRE).unsqueeze(-1), fire_options.expand(B, U, -1),
                                  torch.where((verb == V_SILO).unsqueeze(-1), take(silo), take(b["mask_toggle"]))))
        o_logits = (self.option_head(ctx) * inv_t).masked_fill(~omask, NEG)
        option = _pick(o_logits, lab("uoption"), mode, top_p)
        score(o_logits, option, omask, opt_on, "option", "option")
        ctx = ctx + self.option_embed(option) * opt_on.unsqueeze(-1)
        # count (produce)
        c_log = self.count_head(ctx) * inv_t
        count = _pick(c_log, lab("ucount"), mode, top_p)
        score(c_log, count, torch.ones_like(c_log, dtype=torch.bool), prd, "count", "count")
        # placement of a build: a spot in a map cell, or against one of our buildings
        placing = bld & (bkind < 2)
        anchor_all = b["mask_anchor"] & tokmask                                             # [B,T]
        pmask = torch.stack([torch.ones_like(placing), anchor_all.any(-1, keepdim=True).expand(-1, U)], -1)
        p_log = (self.place_head(ctx) * inv_t).masked_fill(~pmask, NEG)
        place = _pick(p_log, lab("uplace"), mode, top_p)
        score(p_log, place, pmask, given("uplace", placing), "place", "place")
        nxt = placing & (place == 1)
        ctx = ctx + self.place_embed(place) * placing.unsqueeze(-1)
        amask = _safe(anchor_all).unsqueeze(1).expand(-1, U, -1)
        a_logits = (torch.einsum("btd,bud->but", h, self.anchor_q(ctx)) * (scale_d * inv_t)).masked_fill(~amask, NEG)
        anchor = _pick(a_logits, lab("uanchor"), mode, top_p)
        score(a_logits, anchor, amask, given("uanchor", nxt), "place", "anchor")
        ctx = ctx + self.anchor_proj(row_of(h, anchor)) * nxt.unsqueeze(-1)
        # map cell
        cl = given("ucell", needs_cell[verb] & on & ~nxt)
        c_logits = (torch.einsum("bcg,buc->bug", mflat, self.cell_q(ctx)) * scale_c + cell_bias.unsqueeze(1)) * inv_t
        build_cells = b["mask_build_cell"].flatten(2).gather(
            1, bkind.clamp(max=3).unsqueeze(-1).expand(-1, -1, G2))                         # [B,U,G2]
        cmask = _safe(torch.where(bld.unsqueeze(-1), build_cells,
                                  b["mask_move_cell"].flatten(1).unsqueeze(1).expand(-1, U, -1)))
        c_logits = c_logits.masked_fill(~cmask, NEG)
        cell = _pick(c_logits, lab("ucell"), mode, top_p)
        if actions is None and self.commander_greedy:
            cell = torch.where(on & ((verb == V_CMOVE) | (verb == V_CAMOVE)), c_logits.argmax(-1), cell)
        score(c_logits, cell, cmask, cl, "cell", "cell")
        if full:
            guess = c_logits.argmax(-1)
            near = (((guess % GRID) - (cell % GRID)).abs() <= 1) & (((guess // GRID) - (cell // GRID)).abs() <= 1)
            metrics["cell_near"][0] += (near & cl).float().sum(1)
            metrics["cell_near"][1] += cl.float().sum(1)
        ctx = ctx + self.cell_proj(row_of(mflat.transpose(1, 2), cell)) * cl.unsqueeze(-1)
        # the fine pixel inside the cell
        sub_w, fr, i0 = self._fine(h, m, b, ctx, cell, cl & ~(bld & (bkind >= 2)), lab, given, mode, top_p, inv_t,
                                   score, metrics, full)
        if fr is not None:
            ctx = ctx + sub_w[1]
            sub_w = sub_w[0]
        # the spot inside the cell (of its fine pixel, on a map with them)
        sp = cl & placing
        sp_log = self.spot_head(ctx) * inv_t
        spot = _pick(sp_log, lab("uspot"), mode, top_p)
        score(sp_log, spot, torch.ones_like(sp_log, dtype=torch.bool), given("uspot", sp), "place", "spot")
        # target unit
        trow = target_row[verb]
        tg = (trow >= 0) & on
        tmask_row = row_of(_safe(b["mask_target"] & tokmask.unsqueeze(1)), trow.clamp(min=0))  # [B,U,T]
        t_logits = (torch.einsum("btd,bud->but", h, self.tgt_q(ctx)) * (scale_d * inv_t)).masked_fill(~tmask_row, NEG)
        target = _pick(t_logits, lab("utarget"), mode, top_p)
        score(t_logits, target, tmask_row, tg, "target", "target")
        # Per unit, for the factored policy gradient (tools_trainer): a unit's log-probability is its verb's plus
        # its arguments', at its token; `unit_known` marks the units whose choice is scored. Carried in `ent`
        # (keyed, like its entropies) because `metrics` holds [hit, total] pairs only.
        if self.debug and actions is None:
            def top3(logits):
                p_ = F.softmax(logits.float(), -1)
                return p_.topk(min(3, p_.shape[-1]), -1)
            vp, vi = top3(vlog)
            cp, ci = top3(c_logits)
            tp, ti = top3(t_logits)
            bpp, bpi = top3(b_logits)
            self.last_debug = {"vtop_p": vp, "vtop_i": vi, "actor": actor, "idx": idx, "on": on, "verb": verb,
                               "cell_p": cp, "cell_i": ci, "tgt_p": tp, "tgt_i": ti, "bp_p": bpp, "bp_i": bpi,
                               "cell_on": cl, "tgt_on": tg, "bp_on": bp_on}
        ent["unit_logp"] = unit_lp.scatter_add(1, idx, arg_lp * on.float())
        ent["unit_known"] = knownf
        if actions is not None:
            return actions, logp, ent, loss, metrics

        def back(x, o):                         # a [B,U] argument back at its unit's token; -1 where it has none
            return torch.full((B, T), -1, dtype=torch.long, device=dev).scatter_(
                1, idx, torch.where(o, x, torch.full_like(x, -1)))

        out = {"uverb": uverb, "ubp": back(bp, bp_on), "uenh": back(enh, enh_on), "uoption": back(option, opt_on),
               "ucount": back(count, prd), "uplace": back(place, placing), "uanchor": back(anchor, nxt),
               "ucell": back(cell, cl), "uspot": back(spot, sp), "utarget": back(target, tg)}
        if fr is not None:                      # the window pixel back to the cell's own (fz * FINE + fx)
            sx, sz = sub_w % FINE - i0[0].unsqueeze(1), sub_w // FINE - i0[1].unsqueeze(1)
            out["usub"] = back(sz * FINE + sx, fr)
        else:
            out["usub"] = torch.full((B, T), -1, dtype=torch.long, device=dev)
        return out, logp, ent, loss, metrics

    def _fine(self, h, m, b, ctx, cell, cl, lab, given, mode, top_p, inv_t, score, metrics, full):
        """The fine pixel of each unit's chosen cell, on maps with them (b["fine"]: pixels per cell side > 1).
        Returns ((window pixel [B,U], context added [B,U,core]), rows [B,U], (i0x, i0z) [B]) -- the window is
        FINE x FINE pixels of one fine pixel each, the cell's own at i0 .. i0 + n -- or (None, None, None)."""
        fine = b.get("fine")
        if not self.fine_positions or fine is None:
            return None, None, None
        B, U = cell.shape
        dev = h.device
        nx, nz = fine[:, 0].round().long().clamp(1, FINE), fine[:, 1].round().long().clamp(1, FINE)       # [B]
        i0x, i0z = (FINE - nx) // 2, (FINE - nz) // 2
        fr = cl & ((nx > 1) | (nz > 1)).unsqueeze(1)
        fr = given("usub", fr)
        FF = FINE * FINE
        lab_sub = lab("usub")
        if lab_sub is not None:                 # labels are the cell's own pixel: as a window pixel
            s = lab_sub.clamp(min=0)
            lab_sub = torch.where(lab_sub >= 0, (i0z.unsqueeze(1) + s // FINE) * FINE + i0x.unsqueeze(1) + s % FINE,
                                  lab_sub)
        rb, ru = fr.nonzero(as_tuple=True)
        n = rb.shape[0]
        f_logits = torch.zeros(B, U, FF, device=dev)
        fmask = torch.zeros(B, U, FF, dtype=torch.bool, device=dev)
        added = torch.zeros_like(ctx)
        if n:
            # one window per (decision, cell): the units sent to the same cell share it (a battle's attack-move
            # is dozens of units; per unit, 16 k rows of a 640-decision batch held 4 GB of token copies)
            uk, inv = torch.unique(rb * G2 + cell[rb, ru], return_inverse=True)
            ub, uc = uk // G2, uk % G2
            nw = uk.shape[0]
            cx, cz = (uc % GRID).float(), (uc // GRID).float()
            rx, rz = nx[ub].float(), nz[ub].float()
            ox, oz = i0x[ub].float(), i0z[ub].float()
            k = torch.arange(FINE, device=dev, dtype=torch.float32) + 0.5
            # the window's pixel centres as fractions of the map ([-1, 1] for grid_sample)
            px = (cx.unsqueeze(1) + (k - ox.unsqueeze(1)) / rx.unsqueeze(1)) / GRID                       # [w,F] x
            pz = (cz.unsqueeze(1) + (k - oz.unsqueeze(1)) / rz.unsqueeze(1)) / GRID                       # [w,F] z
            grid = torch.stack([px.unsqueeze(1).expand(-1, FINE, -1), pz.unsqueeze(2).expand(-1, -1, FINE)], -1)
            f0 = F.grid_sample(self.fine_map(m)[ub].float(), grid * 2 - 1, mode="bilinear", padding_mode="border",
                               align_corners=False)                                                     # [w,8,F,F]
            # the tokens in the window, at their pixels
            pos = b["tokens"][..., 7:9][ub]                                                              # [w,T,2]
            wx = ((pos[..., 0] * GRID - cx.unsqueeze(1)) * rx.unsqueeze(1) + ox.unsqueeze(1)).floor().long()
            wz = ((pos[..., 1] * GRID - cz.unsqueeze(1)) * rz.unsqueeze(1) + oz.unsqueeze(1)).floor().long()
            inside = b["token_mask"][ub] & (wx >= 0) & (wx < FINE) & (wz >= 0) & (wz < FINE)               # [w,T]
            widx = wz.clamp(0, FINE - 1) * FINE + wx.clamp(0, FINE - 1)
            vals = self.fine_tok(h)[ub].float() * inside.unsqueeze(-1)
            f1 = torch.zeros(nw, FF, 8, device=dev).scatter_add_(1, widx.unsqueeze(-1).expand(-1, -1, 8), vals)
            cnt = torch.zeros(nw, FF, device=dev).scatter_add_(1, widx, inside.float())
            kk = torch.arange(FINE, device=dev)
            own_x = (kk.unsqueeze(0) >= i0x[ub].unsqueeze(1)) & (kk.unsqueeze(0) < (i0x + nx)[ub].unsqueeze(1))
            own_z = (kk.unsqueeze(0) >= i0z[ub].unsqueeze(1)) & (kk.unsqueeze(0) < (i0z + nz)[ub].unsqueeze(1))
            own = own_z.unsqueeze(2) & own_x.unsqueeze(1)                                                # [w,F,F]
            feat = torch.cat([f0, f1.transpose(1, 2).reshape(nw, 8, FINE, FINE),
                              torch.log1p(cnt).view(nw, 1, FINE, FINE), own.float().unsqueeze(1)], 1)
            P = self.fine_net(feat.to(ctx.dtype))                                                       # [w,32,F,F]
            Pf = P.flatten(2)                                                                           # [w,32,FF]
            q = self.fine_q(ctx[rb, ru])                                                                # [n,32]
            lg = (torch.einsum("ncp,nc->np", Pf[inv], q) / math.sqrt(32.0) + self.fine_bias(P).flatten(1)[inv]) * inv_t
            f_logits = f_logits.to(lg.dtype).index_put((rb, ru), lg)
            fmask = fmask.index_put((rb, ru), own.flatten(1)[inv])
        f_logits = f_logits.masked_fill(~fmask, NEG)
        fmask = _safe(fmask)
        sub_w = _pick(f_logits, lab_sub, mode, top_p)
        score(f_logits, sub_w, fmask, fr, "sub", "sub")
        if full:                                # within one pixel of the label
            guess = f_logits.argmax(-1)
            near = (((guess % FINE) - (sub_w % FINE)).abs() <= 1) & (((guess // FINE) - (sub_w // FINE)).abs() <= 1)
            metrics["sub_near"][0] += (near & fr).float().sum(1)
            metrics["sub_near"][1] += fr.float().sum(1)
        if n:
            pv = Pf.transpose(1, 2)[inv, sub_w[rb, ru]]                                                # [n,32]
            added = added.index_put((rb, ru), self.fine_proj(pv).to(added.dtype))
        return (sub_w, added), fr, (i0x, i0z)

    def decode_slots(self, h, m, z, b, actions: dict | None = None, mode: str = "sample", smoothing: float = 0.0,
                     temperature: float = 1.0):
        """Run the MAX_COMMANDS slots. With `actions` it scores them (teacher forcing); otherwise it chooses
        ('sample' or 'greedy'). Every head's logits are divided by `temperature`. Returns (actions, logp [B],
        entropy dict of [B], bc loss [B], metrics)."""
        inv_t = 1.0 / temperature
        top_p = self.act_top_p if actions is None else None
        B, T, d = h.shape
        dev = h.device
        ar = torch.arange(B, device=dev)
        mflat = m.flatten(2)                                   # [B,C,G2]
        cell_bias = self.cell_bias(m).flatten(1)               # [B,G2]
        is_single, is_set, has_option = IS_SINGLE.to(dev), IS_SET.to(dev), HAS_OPTION.to(dev)
        needs_cell, target_row, fire_options = NEEDS_CELL.to(dev), TARGET_ROW.to(dev), FIRE_OPTIONS.to(dev)
        cand_h, enh_h = self.candidates(b)                     # [B,C,d], [B,E,d]
        cand_kind = b["cand_kind"]                             # [B,C]
        state = z
        full = actions is not None               # scoring needs entropies and losses; acting only log-probs
        # a unit may not get the same single-subject verb twice in one decision (two builds on one builder),
        # but may get different ones (a factory told to produce and to rally)
        used = torch.zeros(B, V, T, dtype=torch.bool, device=dev)
        n_build = torch.zeros(B, dtype=torch.long, device=dev)
        active = torch.ones(B, dtype=torch.bool, device=dev)
        logp = torch.zeros(B, device=dev)
        loss = torch.zeros(B, device=dev)
        ent = {k: torch.zeros(B, device=dev) for k in ENTROPY_KEYS}
        out = {k: [] for k in ACTION_KEYS}
        tokmask = b["token_mask"]
        # masks that do not change between slots, prepared once
        subj_all = b["mask_subject"] & tokmask.unsqueeze(1)                 # [B,V,T]
        can_build = b["can_build"] & b["cand_mask"].unsqueeze(1)            # [B,T,C]
        is_mobile = cand_kind == CAND_MOBILE                                # [B,C]
        can_enh = b["can_enh"] & b["enh_mask"].unsqueeze(1)                 # [B,T,E]
        silo = F.pad(b["mask_silo"], (0, N_OPTIONS - len(SILO_KINDS)))      # [B,T,N_OPTIONS]
        toggle = b["mask_toggle"]                                           # [B,T,N_OPTIONS]
        build_all = b["mask_build_cell"].flatten(2)                         # [B,4,G2]
        move_cells = b["mask_move_cell"].flatten(1)                         # [B,G2]
        target_all = _safe(b["mask_target"] & tokmask.unsqueeze(1))         # [B,NR,T]
        anchor_all = b["mask_anchor"] & tokmask                             # [B,T]
        anchor_mask = _safe(anchor_all)
        has_anchor = anchor_all.any(-1)                                     # [B]
        count_mask = torch.ones(B, NC, dtype=torch.bool, device=dev)
        spot_mask = torch.ones(B, NSP, dtype=torch.bool, device=dev)
        scale_d, scale_c = 1.0 / math.sqrt(d), 1.0 / math.sqrt(self.map_ch)

        # agreement with given actions, per argument type (cloning metrics): [correct, total] per sample
        metrics = {k: [torch.zeros(B, device=dev), torch.zeros(B, device=dev)] for k in METRIC_KEYS}

        def score(logits, choice, mask, on, name=None):
            lp, e, ce = _cat_terms(logits, choice, smoothing, mask, full)
            if full and name is not None:
                hit = (logits.argmax(-1) == choice).float()
                metrics[name][0] += hit * on
                metrics[name][1] += on
            return lp * on, (e * on if full else None), (ce * on if full else None)

        def pointer(keys, query, mask):
            return (torch.einsum("bnd,bd->bn", keys, query) * (scale_d * inv_t)).masked_fill(~mask, NEG)

        for k in range(K):
            if k > 0 and not bool(active.any()):
                break                           # every game has ended its turn: later slots add nothing
            lab = (lambda key: actions[key][:, k]) if actions is not None else (lambda key: None)
            given = (lambda key, on: on & (actions[key][:, k] >= 0)) if actions is not None else (lambda key, on: on)
            # verb (after 'end', only 'end')
            # A single-subject verb ends its eligibility when every usable subject has acted.
            # Mask it before picking the verb: _safe below only supplies dummy argument rows.
            remaining = subj_all & ~used
            available = ~is_single.unsqueeze(0) | remaining.any(-1)
            vmask = b["mask_verb"] & available & active.unsqueeze(-1)
            if self.build_budget is not None:
                vmask[:, V_BUILD] &= n_build < self.build_budget
            vmask[:, V_END] |= ~active
            vlog = (self.verb_head(state) * inv_t).masked_fill(~vmask, NEG)
            verb = _pick(vlog, lab("verb"), mode, top_p)
            parts = [("verb", *score(vlog, verb, vmask, active.float(), "verb"))]
            doing = active & (verb != V_END)
            n_build = n_build + (doing & (verb == V_BUILD)).long()
            ctx = state + self.verb_embed(verb)
            # subject: one unit (pointer); a set is a leader (the same pointer) plus the units that join it
            # (include/exclude per eligible unit, given the leader). M28 orders most sets with one unit of
            # hundreds eligible: include/exclude alone learned to choose nobody (recall .003, 2026-10-05)
            slog = torch.einsum("btd,bd->bt", h, self.subj_q(ctx)) * scale_d
            smask = subj_all[ar, verb]
            sgl = is_single[verb] & doing
            st = is_set[verb] & doing
            ptr_mask = _safe(torch.where(sgl.unsqueeze(-1), smask & ~used[ar, verb], smask))
            s_logits = (slog * inv_t).masked_fill(~ptr_mask, NEG)
            given_set = actions["set"][:, k] & smask if actions is not None else None
            s_label = None
            if actions is not None:
                # a set's leader as the policy chose it; labels (and older rollouts) name none: its first member
                first = given_set.int().argmax(-1)
                s_label = torch.where(st & (actions["single"][:, k] < 0), first, actions["single"][:, k])
            s_choice = _pick(s_logits, s_label, mode, top_p)
            ptr_on = (sgl | st).float()
            lp, e, ce = _cat_terms(s_logits, s_choice, smoothing, ptr_mask, full)
            if full:
                hit = (s_logits.argmax(-1) == s_choice).float()
                metrics["subject"][0] += hit * sgl.float()
                metrics["subject"][1] += sgl.float()
                if actions is not None:     # cloning: any member is a right leader
                    has = given_set.any(-1)
                    member = torch.logsumexp(s_logits.masked_fill(~given_set, NEG), -1)
                    any_member = torch.logsumexp(s_logits, -1) - member
                    ce = torch.where(st, torch.where(has, any_member, torch.zeros_like(ce)), ce)
                    lead_hit = given_set.gather(1, s_logits.argmax(-1, keepdim=True)).squeeze(1).float()
                    metrics["set_leader"][0] += lead_hit * st.float()
                    metrics["set_leader"][1] += st.float()
            parts.append(("subject", lp * ptr_on, e * ptr_on if full else None, ce * ptr_on if full else None))
            lead = F.one_hot(s_choice, T).bool() & st.unsqueeze(-1)
            mctx = ctx + self.lead_proj(h[ar, s_choice])
            blog = (torch.einsum("btd,bd->bt", h, self.member_q(mctx)) * scale_d + self.set_bias(mctx)) * inv_t
            mmask = smask & ~lead
            if actions is not None:
                joined = given_set & ~lead
            elif mode == "greedy":
                joined = (blog > 0) & mmask
            else:
                pj = torch.sigmoid(blog)
                if top_p is not None:       # the same cut for joining a set: drop the unlikely tail on both sides
                    tail = (1.0 - top_p)
                    pj = torch.where(pj < tail, torch.zeros_like(pj), torch.where(pj > 1 - tail, torch.ones_like(pj), pj))
                joined = (torch.rand_like(blog) < pj) & mmask
            chosen = joined | lead
            lps_in, lps_out = F.logsigmoid(blog), F.logsigmoid(-blog)
            per = torch.where(joined, lps_in, lps_out).masked_fill(~mmask, 0.0).sum(-1)
            logp = logp + per * st.float()
            if full:
                n_el = mmask.sum(-1).clamp(min=1).float()
                pb = torch.sigmoid(blog)
                bent = -(pb * lps_in + (1 - pb) * lps_out).masked_fill(~mmask, 0.0).sum(-1) / n_el
                ent["subject"] = ent["subject"] + bent * st.float()
                # the set the policy would choose: its own leader, and who joins (given the leader in hand)
                predicted = (((blog > 0) & mmask) | F.one_hot(s_logits.argmax(-1), T).bool()) & smask
                n_all = smask.sum(-1).clamp(min=1).float()
                right = ((predicted == chosen) & smask).float().sum(-1) / n_all
                metrics["set_unit"][0] += right * st.float()
                metrics["set_unit"][1] += st.float()
                hits = (predicted & chosen).sum(-1).float() * st.float()
                metrics["set_recall"][0] += hits
                metrics["set_recall"][1] += chosen.sum(-1).float() * st.float()
                metrics["set_precision"][0] += hits
                metrics["set_precision"][1] += predicted.sum(-1).float() * st.float()
                metrics["set_nonempty"][0] += predicted.any(-1).float() * st.float()
                metrics["set_nonempty"][1] += st.float()
                tgt = joined.float() * (1 - smoothing) + smoothing * 0.5
                if self.bc_set_loss_mode not in ("mean", "balanced"):
                    raise ValueError("Unknown cloning SET loss mode")
                bce = _reduce_set_bc(-(tgt * lps_in + (1 - tgt) * lps_out), joined, mmask,
                                     balanced=self.bc_set_loss_mode == "balanced")
                loss = loss + bce * st.float()
            # who acts, summarised for the rest of the command
            one = F.one_hot(s_choice, T).bool() & sgl.unsqueeze(-1)
            members = torch.where(sgl.unsqueeze(-1), one, chosen & st.unsqueeze(-1)).float()
            actor = torch.einsum("bt,btd->bd", members, h) / members.sum(-1, keepdim=True).clamp(min=1.0)
            ctx = ctx + self.actor_proj(actor)
            used[ar, verb] |= one
            # blueprint (build: a structure the builder can build now; produce: a unit the factory can make;
            # upgrade: which upgrade, when the subject offers several -- a factory's HQ or support factory)
            bld, prd = doing & (verb == V_BUILD), doing & (verb == V_PRODUCE)
            row = can_build[ar, s_choice]                                   # [B,C]
            structures = row & ~is_mobile
            upg = doing & (verb == V_UPGRADE) & structures.any(-1)
            bmask = _safe(torch.where((bld | upg).unsqueeze(-1), structures, row & is_mobile))
            b_logits = pointer(cand_h, self.bp_q(ctx), bmask)
            bp = _pick(b_logits, lab("bp"), mode, top_p)
            bp_on = bld | prd | upg
            parts.append(("blueprint", *score(b_logits, bp, bmask, bp_on.float(), "blueprint")))
            ctx = ctx + self.bp_proj(cand_h[ar, bp]) * bp_on.unsqueeze(-1)
            bkind = cand_kind[ar, bp].clamp(0, CAND_MOBILE)
            # enhancement
            enh_on = doing & (verb == V_ENHANCE)
            emask = _safe(can_enh[ar, s_choice])
            e_logits = pointer(enh_h, self.enh_q(ctx), emask)
            enh = _pick(e_logits, lab("enh"), mode, top_p)
            parts.append(("enhancement", *score(e_logits, enh, emask, enh_on.float(), "enhancement")))
            ctx = ctx + self.enh_proj(enh_h[ar, enh]) * enh_on.unsqueeze(-1)
            # option: fire state, missile kind, or which ability to switch
            opt_on = has_option[verb] & doing
            omask = _safe(torch.where((verb == V_FIRE).unsqueeze(-1), fire_options.expand(B, -1),
                                      torch.where((verb == V_SILO).unsqueeze(-1), silo[ar, s_choice],
                                                  toggle[ar, s_choice])))
            o_logits = (self.option_head(ctx) * inv_t).masked_fill(~omask, NEG)
            option = _pick(o_logits, lab("option"), mode, top_p)
            parts.append(("option", *score(o_logits, option, omask, opt_on.float(), "option")))
            ctx = ctx + self.option_embed(option) * opt_on.unsqueeze(-1)
            # count (produce)
            c_log = self.count_head(ctx) * inv_t
            count = _pick(c_log, lab("count"), mode, top_p)
            parts.append(("count", *score(c_log, count, count_mask, prd.float(), "count")))
            # placement of a build: a spot in a map cell, or against one of our buildings (adjacency); extractors
            # and hydrocarbon plants go on their deposit
            placing = bld & (bkind < 2)
            pmask = torch.stack([torch.ones_like(placing), has_anchor], -1)
            p_log = (self.place_head(ctx) * inv_t).masked_fill(~pmask, NEG)
            place = _pick(p_log, lab("place"), mode, top_p)
            parts.append(("place", *score(p_log, place, pmask, given("place", placing).float(), "place")))
            nxt = placing & (place == 1)
            ctx = ctx + self.place_embed(place) * placing.unsqueeze(-1)
            a_logits = pointer(h, self.anchor_q(ctx), anchor_mask)
            anchor = _pick(a_logits, lab("anchor"), mode, top_p)
            parts.append(("place", *score(a_logits, anchor, anchor_mask, given("anchor", nxt).float(), "anchor")))
            ctx = ctx + self.anchor_proj(h[ar, anchor]) * nxt.unsqueeze(-1)
            # map cell
            cl = given("cell", needs_cell[verb] & doing & ~nxt)       # teacher labels may leave the cell open
            c_logits = (torch.einsum("bcg,bc->bg", mflat, self.cell_q(ctx)) * scale_c + cell_bias) * inv_t
            cmask = _safe(torch.where(bld.unsqueeze(-1), build_all[ar, bkind.clamp(max=3)], move_cells))
            c_logits = c_logits.masked_fill(~cmask, NEG)
            cell = _pick(c_logits, lab("cell"), mode, top_p)
            parts.append(("cell", *score(c_logits, cell, cmask, cl.float(), "cell")))
            if full:
                guess = c_logits.argmax(-1)
                near = (((guess % GRID) - (cell % GRID)).abs() <= 1) & (((guess // GRID) - (cell // GRID)).abs() <= 1)
                metrics["cell_near"][0] += near.float() * cl.float()
                metrics["cell_near"][1] += cl.float()
            cell_feat = mflat.gather(2, cell.view(B, 1, 1).expand(-1, self.map_ch, 1)).squeeze(-1)
            ctx = ctx + self.cell_proj(cell_feat) * cl.unsqueeze(-1)
            # the spot inside the cell (a quarter cell each way)
            sp = cl & placing
            sp_log = self.spot_head(ctx) * inv_t
            spot = _pick(sp_log, lab("spot"), mode, top_p)
            parts.append(("place", *score(sp_log, spot, spot_mask, given("spot", sp).float(), "spot")))
            # target unit
            trow = target_row[verb]
            tg = (trow >= 0) & doing
            tmask_row = target_all[ar, trow.clamp(min=0)]
            t_logits = pointer(h, self.tgt_q(ctx), tmask_row)
            target = _pick(t_logits, lab("target"), mode, top_p)
            parts.append(("target", *score(t_logits, target, tmask_row, tg.float(), "target")))
            ctx = ctx + self.tgt_proj(h[ar, target]) * tg.unsqueeze(-1)
            for name, lp, e, ce in parts:
                logp = logp + lp
                if full:
                    ent[name] = ent[name] + e
                    loss = loss + ce
            # the next slot knows this command
            state = self.slot_cell(ctx, state)
            if actions is None:
                none = lambda x, on: torch.where(on, x, torch.full_like(x, -1))  # noqa: E731
                out["verb"].append(torch.where(active, verb, torch.full_like(verb, V_END)))
                out["single"].append(none(s_choice, sgl | st))      # a set: its leader
                out["set"].append(chosen & st.unsqueeze(-1))
                out["bp"].append(none(bp, bp_on))
                out["enh"].append(none(enh, enh_on))
                out["option"].append(none(option, opt_on))
                out["count"].append(none(count, prd))
                out["place"].append(none(place, placing))
                out["anchor"].append(none(anchor, nxt))
                out["cell"].append(none(cell, cl))
                out["spot"].append(none(spot, sp))
                out["target"].append(none(target, tg))
            active = doing
        if actions is None:
            while len(out["verb"]) < K:         # pad the slots skipped after everyone ended
                out["verb"].append(torch.full((B,), V_END, dtype=torch.long, device=dev))
                for key in ACTION_KEYS:
                    if key not in ("verb", "set"):
                        out[key].append(torch.full((B,), -1, dtype=torch.long, device=dev))
                out["set"].append(torch.zeros(B, T, dtype=torch.bool, device=dev))
        chosen_actions = {k: torch.stack(v, 1) for k, v in out.items()} if actions is None else actions
        return chosen_actions, logp, ent, loss, metrics

    # ------------------------------------------------------------------ public API
    def initial_state(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch, self.core_dim, device=device)
        return z, z.clone()

    def encode_any(self, b: dict):
        """`encode`, in bf16 when `amp` is on (outputs back in float32); on a CPU too (a player machine's Zen 4 has
        bf16 instructions)."""
        if self.amp:
            with torch.autocast("cuda" if b["tokens"].is_cuda else "cpu", dtype=torch.bfloat16):
                h, m, core_in = self.encode(b)
            return h.float(), m.float(), core_in.float()
        return self.encode(b)

    def forward_step(self, b: dict, state, mode: str = "sample", temperature: float = 1.0):
        """One decision for a batch of games: (actions, logp, value, new_state)."""
        self.transformer.layers[-1].capture_attn = self.debug
        self.last_debug = None
        h, m, core_in = self.encode_any(b)
        z, new_state = self.lstm(core_in.unsqueeze(1), state)
        z = z.squeeze(1)
        actions, logp, ent, _, _ = self.decode(h, m, z, b, None, mode, temperature=temperature)
        # the units decoder's per-unit log-probabilities of this decision ([B,T]; None for the slots decoder)
        self.last_unit_logp = ent.get("unit_logp")
        value = self.value_of(z, b)
        if self.debug and self.last_debug is not None:
            self.last_debug["value"] = value
            self.last_debug["local_value"] = self.local_value(m).squeeze(1)                    # [B,G,G]
            attn = getattr(self.transformer.layers[-1], "last_attn", None)
            if attn is not None:
                self.last_debug["attn"] = attn
        return actions, logp, value, new_state

    def core(self, b: dict, state, seq_len: int, starts: torch.Tensor | None = None, return_state: bool = False):
        """Encoder and memory for N = n_seq * seq_len observations laid out sequence-major. `starts` [N] marks
        episode starts (the memory is reset there). Returns (h, m, z): what `score` and the heads need; with
        `return_state`, also the memory after each sequence's last step (to carry into the next chunk)."""
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            h, m, core_in = torch.utils.checkpoint.checkpoint(self.encode_any, b, use_reentrant=False)
        else:
            h, m, core_in = self.encode_any(b)
        N = core_in.shape[0]
        n_seq = N // seq_len
        x = core_in.view(n_seq, seq_len, -1)
        hs, cs = state
        zs = []
        for t in range(seq_len):
            if starts is not None:
                keep = (~starts.view(n_seq, seq_len)[:, t]).float().view(1, n_seq, 1)
                hs, cs = hs * keep, cs * keep
            z, (hs, cs) = self.lstm(x[:, t:t + 1], (hs, cs))
            zs.append(z)
        if return_state:
            return h, m, torch.cat(zs, 1).reshape(N, -1), (hs, cs)
        return h, m, torch.cat(zs, 1).reshape(N, -1)

    def score(self, h, m, z, b: dict, actions: dict, smoothing: float = 0.0, temperature: float = 1.0):
        """Score given actions on a `core` output: (logp, entropy dict, bc loss), all [N]; the agreement per
        argument type is left in `last_metrics`."""
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            _, logp, ent, loss, metrics = torch.utils.checkpoint.checkpoint(
                self.decode, h, m, z, b, actions, "sample", smoothing, temperature, use_reentrant=False)
        else:
            _, logp, ent, loss, metrics = self.decode(h, m, z, b, actions, smoothing=smoothing,
                                                      temperature=temperature)
        self.last_metrics = metrics
        return logp, ent, loss

    def value_of(self, z, b: dict):
        """The critic: the memory and the privileged inputs, plus the opponent's part when the batch says who it
        is (training; a game being played has no "opponent" and its value is the value head's alone)."""
        value = self.value_head(torch.cat([z, b["privileged"]], -1)).squeeze(-1)
        if "opponent" in b:
            value = value + self.opp_value(torch.cat([z, b["opponent"].to(z.dtype)], -1)).squeeze(-1)
        return value

    def heads(self, z, b: dict):
        """Value (with the privileged inputs) and belief from the memory."""
        return self.value_of(z, b), self.belief_head(z)

    def evaluate(self, b: dict, actions: dict, state, seq_len: int, starts: torch.Tensor | None = None,
                 smoothing: float = 0.0, temperature: float = 1.0):
        """Score given actions for N = n_seq * seq_len observations laid out sequence-major. `starts` [N] marks
        episode starts (the memory is reset there). Returns logp, entropy dict, value, belief, bc loss (all [N])."""
        h, m, z = self.core(b, state, seq_len, starts)
        logp, ent, loss = self.score(h, m, z, b, actions, smoothing, temperature)
        value, belief = self.heads(z, b)
        return logp, ent, value, belief, loss


# Behaviour switches of a trained network that are not weights: a checkpoint saves them ("model_options") and
# every loader applies them, so a network always runs as it was trained. Missing: False (older checkpoints).
MODEL_OPTIONS = ("verbs_after_attention", "fine_positions")


def model_options(net: "ToolsNet") -> dict:
    return {k: bool(getattr(net, k, False)) for k in MODEL_OPTIONS}


def apply_model_options(net: "ToolsNet", state: dict | None) -> dict:
    options = {k: bool(((state or {}).get("model_options") or {}).get(k, False)) for k in MODEL_OPTIONS}
    for k, v in options.items():
        setattr(net, k, v)
    return options


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------- batching
OPPONENT_FEATURES = ("adaptive", "rush", "tech", "scripted_other", "m28", "m28_cheat", "cheat_multiplier", "self")
N_OPPONENT = len(OPPONENT_FEATURES)


def opponent_features(ai: str) -> np.ndarray:
    """The critic's opponent input from a game's opponent name ("adaptive", "m28aicheat@0.6", "m28ai",
    "self@rl_089", ...)."""
    f = np.zeros(N_OPPONENT, np.float32)
    ai = str(ai or "")
    if ai.startswith("self@"):
        f[7] = 1.0
    elif ai.startswith("m28"):
        f[4] = 1.0
        if "@" in ai:
            f[5] = 1.0
            try:
                f[6] = float(ai.split("@", 1)[1].split(":")[0])
            except ValueError:
                pass
    elif ai in ("adaptive", "rush", "tech"):
        f[("adaptive", "rush", "tech").index(ai)] = 1.0
    else:
        f[3] = 1.0
    return f


def opponent_of_recording(source_name: str) -> str:
    """A recording's opponent from its name, <uid>_<opponent>_<seed>.jsonl.gz (the opponent may hold '_')."""
    parts = str(source_name).removesuffix(".jsonl.gz").split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else ""


def collate(observations: list[dict], device) -> dict:
    """Pad a list of ToolsObserver outputs (numpy) to one batch of tensors: tokens to the longest token list,
    candidates to the longest candidate list (at least one of each, so every softmax has a column)."""
    B = len(observations)
    T = max(1, max(len(o["tokens"]) for o in observations))
    C = max(1, max(len(o["cand_bp"]) for o in observations))
    E = max(1, max(len(o["enh_idx"]) for o in observations))
    out = {}

    def pad(key, n, axis):
        first = observations[0][key]
        shape = list(first.shape)
        shape[axis] = n
        arr = np.zeros((B, *shape), first.dtype)
        for i, o in enumerate(observations):
            a = o[key]
            if axis == 0:
                arr[i, : len(a)] = a
            else:
                arr[i, :, : a.shape[1]] = a
        return arr

    for key in TOKEN_KEYS:
        out[key] = pad(key, T, 0)
    for key in TOKEN_T_KEYS:
        out[key] = pad(key, T, 1)
    for key in CAND_KEYS:
        out[key] = pad(key, C, 0)
    for key in ENH_KEYS:
        out[key] = pad(key, E, 0)
    for key in STEP_KEYS:
        out[key] = np.stack([o[key] for o in observations])
    for key, n in zip(PAIR_KEYS, (C, E)):
        dense = np.zeros((B, T, n), bool)
        for i, o in enumerate(observations):
            p = o[key]
            if len(p):
                dense[i, p[:, 0], p[:, 1]] = True
        out[key] = dense
    for name, key, n in (("token_mask", "tokens", T), ("cand_mask", "cand_bp", C), ("enh_mask", "enh_idx", E)):
        mask = np.zeros((B, n), bool)
        for i, o in enumerate(observations):
            mask[i, : len(o[key])] = True
        out[name] = mask
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


def empty_labels(n_tokens: int) -> dict:
    lab = {k: np.full(K, -1, np.int64) for k in ACTION_KEYS if k != "set"}
    lab["verb"][:] = V_END
    lab["set"] = np.zeros((K, n_tokens), bool)
    return lab


def upgrade_options(x: dict) -> dict[int, set]:
    """token -> the structure candidates it can build although it is not a mobile builder: a factory's upgrades
    (what the network's upgrade verb points at)."""
    cached = x.get("_upgrade_options")
    if cached is not None:
        return cached
    kinds, mobile = x["cand_kind"], x["mask_subject"][VERBS.index("build")]
    out: dict[int, set] = {}
    pairs = np.asarray(x["can_build"], np.int64).reshape(-1, 2)
    if len(pairs):
        keep = (kinds[pairs[:, 1]] != CAND_MOBILE) & ~mobile[pairs[:, 0]]
        for i, j in pairs[keep].tolist():
            out.setdefault(i, set()).add(j)
    x["_upgrade_options"] = out                 # asked twice per decision while encoding; the same answer
    return out


def commands_to_indices(commands: list[dict], x: dict) -> dict:
    """fa_env.tools commands -> per-slot index labels for observation `x` (a ToolsObserver output). Commands whose
    subject, blueprint, enhancement or target the observation does not have are dropped; slots after the last
    command are 'end'."""
    ids = x["ids"]
    index = {u: i for i, u in enumerate(ids) if u is not None}
    cand = {b: j for j, b in enumerate(x.get("cand_ids") or [])}
    enh = {(int(i), x["enh_names"][j]): int(j) for i, j in x.get("can_enh", [])}
    upgrades = upgrade_options(x)
    lab = empty_labels(len(ids))
    k = 0
    for c in commands:
        if k >= K:
            break
        verb = c["verb"]
        subj = [index[u] for u in c.get("subject") or [] if u in index]
        if not subj:
            continue
        one = {}
        if verb in TARGET_VERBS:
            if c.get("target") not in index:
                continue
            one["target"] = index[c["target"]]
        if verb in ("build", "produce") or (verb == "upgrade" and subj[0] in upgrades):
            j = cand.get(c.get("bp"))
            if j is None or (verb == "upgrade" and j not in upgrades[subj[0]]):
                continue
            one["bp"] = j
        if verb == "enhance":
            j = enh.get((subj[0], c.get("enhancement")))
            if j is None:
                continue
            one["enh"] = j
        if verb in OPTION_VERBS:
            one["option"] = int(c.get("option") or 0)
        nxt = False
        if verb == "build" and int(x["cand_kind"][one["bp"]]) < 2:       # placed, not on a deposit
            nxt = c.get("place") == "next"
            one["place"] = 1 if nxt else 0
            if nxt and c.get("anchor") in index:
                one["anchor"] = index[c["anchor"]]
            if not nxt and c.get("spot") is not None:
                one["spot"] = c["spot"][1] * SPOTS + c["spot"][0]
        if verb == "produce":
            one["count"] = COUNTS.index(c["count"]) if c.get("count") in COUNTS else 0
        if verb in CELL_VERBS and c.get("cell") is not None and not nxt:
            gx, gz = c["cell"]
            one["cell"] = gz * GRID + gx
        lab["verb"][k] = VERBS.index(verb)
        if verb in SINGLE_SUBJECT:
            lab["single"][k] = subj[0]
        else:
            lab["set"][k, subj] = True
        for key, value in one.items():
            lab[key][k] = value
        k += 1
    return lab


def empty_unit_labels(n_tokens: int) -> dict:
    lab = {k: np.full(n_tokens, -1, np.int64) for k in UNIT_KEYS}
    lab["uverb"][:] = 0
    return lab


def unit_actions_to_commands(act: dict, i: int, x: dict) -> list[dict]:
    """Row `i` of a batch of per-unit actions (UNIT_KEYS, numpy) -> fa_env.tools commands for observation `x`:
    units given the same set verb with the same arguments become one command, in the canonical verb order."""
    from fa_env.tools import VERB_ORDER
    ids = x["ids"]
    uverb = act["uverb"][i][: len(ids)]
    out, groups = [], {}
    for t in np.flatnonzero(uverb > 0):
        if ids[t] is None:
            continue
        one = {"verb": np.array([[uverb[t]]]), "single": np.array([[t]]), "set": np.zeros((1, 1, len(ids)), bool)}
        one["set"][0, 0, t] = True
        one.update({key[1:]: np.array([[act[key][i][t]]]) for key in UNIT_KEYS if key != "uverb"})
        for c in indices_to_commands(one, 0, x, slots=1):
            if c["verb"] in SET_SUBJECT:
                key = (c["verb"], c.get("cell"), c.get("sub"), c.get("target"), c.get("option"))
                if key in groups:
                    groups[key]["subject"].append(ids[t])
                    continue
                groups[key] = c
            out.append(c)
    out.sort(key=lambda c: VERB_ORDER.get(c["verb"], 99))
    return out


def action_keys(decoder: str) -> tuple:
    return UNIT_KEYS if decoder == "units" else ACTION_KEYS


def action_row(act: dict, i: int, n_tokens: int, decoder: str) -> dict:
    """Row `i` of a batch of chosen actions (numpy), cut to its own tokens: what a rollout stores per decision."""
    if decoder == "units":
        return {k: act[k][i][:n_tokens] for k in UNIT_KEYS}
    return {k: (act[k][i][:, :n_tokens] if k == "set" else act[k][i]) for k in ACTION_KEYS}


def stack_actions(rows: list[dict], n_tokens: int, decoder: str) -> dict:
    """Stored rows (or labels) padded to `n_tokens` and stacked: numpy arrays for `decode(..., actions=...)`."""
    out = {}
    if decoder == "units":
        for key in UNIT_KEYS:
            arr = np.full((len(rows), n_tokens), 0 if key == "uverb" else -1, np.int64)
            for i, r in enumerate(rows):
                arr[i, : len(r[key])] = r[key]
            out[key] = arr
        return out
    for key in ACTION_KEYS:
        if key == "set":
            arr = np.zeros((len(rows), K, n_tokens), bool)
            for i, r in enumerate(rows):
                arr[i, :, : r["set"].shape[1]] = r["set"]
        else:
            arr = np.stack([r[key] for r in rows])
        out[key] = arr
    return out


def actions_to_commands(act: dict, i: int, x: dict, decoder: str) -> list[dict]:
    return unit_actions_to_commands(act, i, x) if decoder == "units" else indices_to_commands(act, i, x)


def indices_to_commands(act: dict, i: int, x: dict, slots: int = K) -> list[dict]:
    """Row `i` of a batch of chosen actions (tensors moved to numpy) -> fa_env.tools commands, for observation `x`."""
    ids, cand_ids, enh_names = x["ids"], x.get("cand_ids") or [], x.get("enh_names") or []
    tok = lambda j: ids[j] if 0 <= j < len(ids) else None  # noqa: E731
    out = []
    for k in range(slots):
        v = int(act["verb"][i, k])
        if v == V_END:
            break
        verb = VERBS[v]
        if verb in SINGLE_SUBJECT:
            j = int(act["single"][i, k])
            subject = [ids[j]] if tok(j) is not None else []
        else:
            sel = np.flatnonzero(act["set"][i, k][: len(ids)])
            subject = [ids[j] for j in sel if ids[j] is not None]
        c = {"verb": verb, "subject": subject}
        nxt = False
        if verb in ("build", "produce"):
            j = int(act["bp"][i, k])
            if not 0 <= j < len(cand_ids):
                continue
            c["bp"] = cand_ids[j]
            if verb == "build" and int(x["cand_kind"][j]) < 2:
                nxt = int(act["place"][i, k]) == 1
                c["place"] = PLACES[1 if nxt else 0]
                if nxt:
                    c["anchor"] = tok(int(act["anchor"][i, k]))
                else:
                    s = int(act["spot"][i, k])
                    c["spot"] = (s % SPOTS, s // SPOTS) if s >= 0 else None
            if verb == "produce":
                c["count"] = COUNTS[int(act["count"][i, k])]
        if verb == "upgrade":
            j = int(act["bp"][i, k])
            if 0 <= j < len(cand_ids):
                c["bp"] = cand_ids[j]
        if verb == "enhance":
            j = int(act["enh"][i, k])
            if not 0 <= j < len(enh_names):
                continue
            c["enhancement"] = enh_names[j]
        if verb in OPTION_VERBS:
            c["option"] = int(act["option"][i, k])
        if verb in CELL_VERBS and not nxt:
            cell = int(act["cell"][i, k])
            c["cell"] = (cell % GRID, cell // GRID)
            s = int(act["sub"][i, k]) if "sub" in act else -1
            if s >= 0:
                c["sub"] = (s % FINE, s // FINE)
        if verb in TARGET_VERBS:
            c["target"] = tok(int(act["target"][i, k]))
        out.append(c)
    return out
