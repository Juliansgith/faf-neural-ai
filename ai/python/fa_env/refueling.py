"""Own-aircraft telemetry and typed loading compatibility; no enemy information."""
import math
import numpy as np

LAND_CARGO, AIRCRAFT = 1, 2
N_REFUEL_FEATURES = 5  # known fuel, fuel ratio, use duration, staging platform, air transport


def observation_fields(rows, blueprints, own_type=0):
    features = np.zeros((len(rows), N_REFUEL_FEATURES), np.float32)
    subjects = np.zeros(len(rows), np.int64)
    targets = np.zeros(len(rows), np.int64)
    for i, (kind, unit) in enumerate(rows):
        if kind != own_type:
            continue
        cats = set(blueprints.get(unit.get('bp'), {}).get('categories') or [])
        if 'AIR' in cats and 'fuel' in unit:
            try:
                fuel, duration = float(unit['fuel']), float(unit.get('fuel_use_time') or 0)
            except (TypeError, ValueError):
                fuel, duration = math.nan, math.nan
            if math.isfinite(fuel) and math.isfinite(duration) and duration > 0:
                features[i, :3] = (1., min(1., max(0., fuel)), math.log1p(duration) / 8.)
        fraction = 1. if unit.get('frac') is None else float(unit['frac'])
        if fraction < 1:
            continue
        if 'MOBILE' in cats and not {'COMMAND', 'EXPERIMENTAL'} & cats:
            if 'AIR' in cats and 'CANNOTUSEAIRSTAGING' not in cats:
                subjects[i] = AIRCRAFT
            elif 'LAND' in cats and 'AIR' not in cats:
                subjects[i] = LAND_CARGO
        if {'AIR', 'TRANSPORTATION'} <= cats:
            targets[i] |= LAND_CARGO
            features[i, 4] = 1.
        if 'AIRSTAGINGPLATFORM' in cats:
            targets[i] |= AIRCRAFT
            features[i, 3] = 1.
    return dict(token_refueling=features, load_subject_class=subjects, load_target_class=targets)


def enabled_observation(x):
    """An alternate label view. Leave the legacy observation and labels untouched."""
    from .tools import VERBS, TARGET_ROWS
    subject, target = x['load_subject_class'], x['load_target_class']
    eligible = (((subject == LAND_CARGO) & ((target & LAND_CARGO) != 0).any()) |
                ((subject == AIRCRAFT) & ((target & AIRCRAFT) != 0).any()))
    out = {**x, 'mask_subject': x['mask_subject'].copy(), 'mask_target': x['mask_target'].copy(),
           'mask_verb': x['mask_verb'].copy(), 'refueling_enabled': True}
    load = VERBS.index('load')
    out['mask_subject'][load] = eligible
    out['mask_target'][TARGET_ROWS.index('transport')] = target != 0
    out['mask_verb'][load] = eligible.any()
    return out
