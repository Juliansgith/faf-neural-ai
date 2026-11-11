"""Versioned behavior probabilities: what a recorded log-probability means. The hard 2 % floor samples from one
distribution and scores another; the smooth family samples and scores the same one."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

LEGACY = 'units-legacy-unfloored-v1'
SMOOTH = 'units-smooth-verb-v1-eps.01-k8-tau.02'
SMOOTH_UPGRADE_FREE = SMOOTH + '-upgrade-free'
SMOOTH_FAMILY = {SMOOTH, SMOOTH_UPGRADE_FREE}
SUPPORTED = {LEGACY, SMOOTH, SMOOTH_UPGRADE_FREE}


def semantics(net) -> str:
    if not getattr(net,'smooth_verb_floor',False):
        return LEGACY
    return SMOOTH_UPGRADE_FREE if getattr(net,'upgrade_floor_free',False) else SMOOTH


def smooth_log_probabilities(logits, mask, free_verbs=()):
    """q proportional to p * [.01 + .99 sigmoid(8 log(p/.02))]; END and `free_verbs` retained."""
    lp = F.log_softmax(logits.masked_fill(~mask,-1e9),-1)
    retention = torch.logaddexp(torch.full_like(lp,math.log(.01)),
                               math.log(.99)+F.logsigmoid(8*(lp-math.log(.02))))
    kept = torch.zeros(lp.shape[-1],dtype=torch.bool,device=lp.device)
    kept[[0,*free_verbs]] = True
    retention = torch.where(kept,torch.zeros_like(retention),retention)
    return F.log_softmax((lp+retention).masked_fill(~mask,-1e9),-1).masked_fill(~mask,-1e9)


def validate_acting_settings(net, mode, temperature):
    if semantics(net) not in SMOOTH_FAMILY:
        return
    if (net.decoder != 'units' or net.act_min_p != .02 or net.act_min_p_verbs is not None
            or net.act_top_p is not None or net.commander_greedy or mode != 'sample'
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Smooth behavior requires units/sample, default verb threshold, no top-p or greedy commander')


def recording_semantics(policies) -> str:
    """Unmarked historical records are legacy; explicit unknown/mixed tags fail."""
    tags, temperatures = set(), set()
    for policy in policies:
        if not policy or not isinstance(policy.get('action'),dict):
            continue
        tag = policy.get('logp_semantics',LEGACY)
        if not isinstance(tag,str) or tag not in SUPPORTED:
            raise ValueError(f'Unknown recorded probability semantics: {tag!r}')
        if tag in SMOOTH_FAMILY:
            temperature = policy.get('temperature')
            if (policy.get('mode') != 'sample' or not isinstance(temperature,(int,float))
                    or isinstance(temperature,bool) or not math.isfinite(temperature) or temperature<=0):
                raise ValueError('Smooth recording requires explicit sample mode and finite positive temperature')
            logp = policy.get('logp')
            if (not isinstance(logp, (int, float)) or isinstance(logp, bool)
                    or not math.isfinite(logp)):
                raise ValueError('Smooth recording has nonfinite probability')
            unit_logp = policy.get('unit_logp')
            if (not isinstance(unit_logp, list) or not unit_logp
                    or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                           or not math.isfinite(v) for v in unit_logp)):
                raise ValueError('Smooth recording requires finite per-unit probabilities')
            temperatures.add(float(temperature))
        tags.add(tag)
    if len(tags)>1:
        raise ValueError('A recording mixes incompatible probability semantics')
    if len(temperatures)>1:
        raise ValueError('A smooth recording mixes sampling temperatures')
    return next(iter(tags),LEGACY)


def recording_temperature(policies):
    """Only the explicit smooth protocol guarantees a single recorded temperature."""
    if recording_semantics(policies) not in SMOOTH_FAMILY:
        return None
    return next(float(p['temperature']) for p in policies
                if p and isinstance(p.get('action'), dict))


def cached_semantics(game) -> str:
    value = game.get('pol_logp_semantics')
    tag = LEGACY if value is None else str(np.asarray(value).item())
    if tag not in SUPPORTED:
        raise ValueError(f'Unknown cached probability semantics: {tag!r}')
    return tag


def compatible_ppo_games(net, games, candidates, temperature=.5):
    expected = semantics(net)
    accepted,rejected = [],[]
    for index in candidates:
        game = games.games[index]
        tag = cached_semantics(game)
        matches = tag == expected
        if tag in SMOOTH_FAMILY:
            recorded = game.get('pol_temperature')
            if recorded is None:
                raise ValueError('Smooth cache has no recorded sampling temperature')
            recorded = float(np.asarray(recorded).item())
            if not math.isfinite(recorded) or recorded <= 0:
                raise ValueError('Smooth cache has an invalid sampling temperature')
            matches = matches and recorded == temperature
        (accepted if matches else rejected).append(index)
    return accepted,rejected


def align_reference_distribution(net, reference):
    # The anchor uses the original weights under the same continuous family;
    # this is not a claim about the original hard-floor behavior's exact KL.
    reference.smooth_verb_floor = bool(getattr(net,'smooth_verb_floor',False))
    reference.upgrade_floor_free = bool(getattr(net,'upgrade_floor_free',False))
