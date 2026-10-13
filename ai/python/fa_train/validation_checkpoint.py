"""Keep completed optimizer work recoverable before validation can fail."""
from pathlib import Path
import torch


def save_unvalidated_checkpoint(path, net, optimizer, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve existing recovery checkpoint: {path}')
    weights = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if not all(torch.isfinite(v).all().item() for v in weights.values()):
        raise RuntimeError('Nonfinite candidate tensors')
    payload = {**metadata, 'model': weights, 'optimizer': optimizer.state_dict(),
               'validation_status': 'pending', 'strength': 'unproven',
               'resume_note': 'Recoverable model/optimizer; prefetch prevents bit-exact RNG replay. Not a selected checkpoint.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()
    return weights


def checkpoint_then_validate(path, net, optimizer, metadata, validate):
    weights = save_unvalidated_checkpoint(path, net, optimizer, metadata)
    return weights, validate()
