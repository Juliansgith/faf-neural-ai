"""Optional queued factory upgrades, derived solely from existing own-unit fields."""
import numpy as np


def label_prefix(options):
    return ('rf_' if getattr(options, 'air_refueling', False) else '') + (
        'fu_' if getattr(options, 'queued_factory_upgrades', False) else '')


def enabled_observation(x):
    from .tools import VERBS
    from .tools_obs import CAND_MOBILE
    legal = np.zeros(len(x['tokens']), bool)
    pairs = np.asarray(x['can_build'], dtype=np.int64).reshape(-1, 2)
    if len(pairs):
        legal[pairs[x['cand_kind'][pairs[:, 1]] != CAND_MOBILE, 0]] = True
    # Queue length is stored divided by ten; tolerate only its float16 rounding.
    eligible = (x['mask_subject'][VERBS.index('rally')] & (x['token_type'] == 0)
                & (x['tokens'][:, 14] >= 1) & (x['token_order'] == 8)
                & (np.abs(x['tokens'][:, 17] - .1) < .001) & legal)
    out = {**x, 'mask_subject': x['mask_subject'].copy(), 'mask_verb': x['mask_verb'].copy()}
    upgrade = VERBS.index('upgrade')
    out['mask_subject'][upgrade] |= eligible
    out['mask_verb'][upgrade] = out['mask_subject'][upgrade].any()
    return out


def enabled_batch(b):
    from .tools import VERBS
    from .tools_obs import CAND_MOBILE
    legal = (b['can_build'] & b['cand_mask'].unsqueeze(1)
             & (b['cand_kind'] != CAND_MOBILE).unsqueeze(1)).any(-1)
    eligible = (b['mask_subject'][:, VERBS.index('rally')] & b['token_mask']
                & (b['token_type'] == 0) & (b['tokens'][..., 14] >= 1)
                & (b['token_order'] == 8) & ((b['tokens'][..., 17] - .1).abs() < .001) & legal)
    out = {**b, 'mask_subject': b['mask_subject'].clone(), 'mask_verb': b['mask_verb'].clone()}
    upgrade = VERBS.index('upgrade')
    out['mask_subject'][:, upgrade] |= eligible
    out['mask_verb'][:, upgrade] = out['mask_subject'][:, upgrade].any(-1)
    return out
