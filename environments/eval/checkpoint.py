"""Verify and inspect a scoring checkpoint without executing pickle globals."""
from pathlib import Path
from environments.eval.common import digest, read, require


def read_model_payload(path):
    import torch
    path = Path(path)
    item = read(path / 'complete.json')['files']['model.pt']
    require((path / 'model.pt').stat().st_size == item['bytes'] and
            digest(path / 'model.pt') == item['sha256'], 'checkpoint checksum mismatch: model.pt')
    # Trainer checkpoints record torch.__version__, a str subclass that runs no code when rebuilt.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
