"""CPU replay of complete recorded games; no acting-policy changes or GPU use.

Scores a checkpoint against recorded scores, optionally compares another frozen
checkpoint on exactly the same observations and recurrent histories. This is a
diagnostic, not a substitute for ordinary-start performance tests.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from fa_train.learn_rl import load_net, policy_probe
from fa_train.tools_bc import Games
from examples.tools_eval import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--compare', type=Path)
    ap.add_argument('--cache', type=Path, action='append', required=True)
    ap.add_argument('--sources', type=Path, help='JSON list of [source_root, source_name]')
    ap.add_argument('--games', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    sources = {tuple(x) for x in json.loads(a.sources.read_text(encoding='utf-8'))} if a.sources else None
    games = Games(a.cache, newest=a.games, sources=sources)
    games.units = True
    ids = [i for i, g in enumerate(games.games) if 'pol_has' in g and g['pol_has'].any()]
    if not ids:
        raise SystemExit('No usable current-format recorded games')
    lengths = {i: int(games.lengths[i]) for i in ids}
    args = SimpleNamespace(micro=1, length=16, temperature=.5, stop_file=None)
    net, _ = load_net(a.checkpoint, 'cpu')
    print(f'Replaying {len(ids)} complete games ({sum(lengths.values())} decisions) on CPU', flush=True)
    baseline, payload = policy_probe(net, games, ids, lengths, args, 'cpu', None)
    baseline['sample_k3_to_recorded_scores'] = baseline.pop('sample_k3_to_init')
    out = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha256(a.checkpoint), baseline=baseline,
               source_manifests=[games.games[i]['_cache_manifest'] for i in ids])
    if a.compare:
        other, _ = load_net(a.compare, 'cpu')
        comparison, _ = policy_probe(other, games, ids, lengths, args, 'cpu',
                                     {i: payload[i]['logp'] for i in ids}, before=payload)
        comparison['verb_mode_change_fraction'] = comparison.pop('verb_mode_changed_by_sil')
        comparison.pop('sample_k3_from_sil')
        out.update(compare=str(a.compare), compare_sha256=sha256(a.compare), comparison=comparison)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'source_manifests'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
