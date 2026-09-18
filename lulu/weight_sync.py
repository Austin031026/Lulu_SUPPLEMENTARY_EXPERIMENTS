"""Load a committed full Student checkpoint without rebuilding resident models."""
from __future__ import annotations
import json
from pathlib import Path


def checkpoint_tensors(path):
    from safetensors import safe_open
    path = Path(path)
    index = path/'model.safetensors.index.json'
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    elif (path/'model.safetensors').is_file():
        files = ['model.safetensors']
    else:
        raise FileNotFoundError(f'No full safetensors checkpoint: {path}')
    for name in files:
        if Path(name).name != name:
            raise ValueError('Checkpoint shard must be a direct filename')
        with safe_open(path/name, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                yield key, stream.get_tensor(key)


def sync_hindsight_checkpoint(model, path, round_index):
    import torch
    state = json.loads((Path(path)/'lulu_state.json').read_text())
    if state['completed_rounds'] != round_index:
        raise ValueError('Hindsight checkpoint belongs to a different round')
    target = model.state_dict()
    loaded = set()
    with torch.no_grad():
        for name, tensor in checkpoint_tensors(path):
            if name not in target or target[name].shape != tensor.shape or target[name].dtype != tensor.dtype:
                raise ValueError(f'Checkpoint parameter mismatch: {name}')
            target[name].copy_(tensor)
            loaded.add(name)
    # HF safetensors may omit aliased tied weights. Verify shared storage aliases.
    missing = set(target)-loaded
    pointers = {target[name].data_ptr() for name in loaded}
    if any(target[name].data_ptr() not in pointers for name in missing):
        raise ValueError(f'Incomplete full Student snapshot: {sorted(missing)}')
    model.requires_grad_(False).eval()


def sync_vllm_checkpoint(worker, path):
    """vLLM collective_rpc callable; stream shards through its native weight loader."""
    import torch
    with torch.no_grad():
        loaded = worker.model_runner.model.load_weights(checkpoint_tensors(path))
    required = set(dict(worker.model_runner.model.named_parameters()))
    if loaded is None or required-set(loaded):
        raise RuntimeError(f'vLLM full-weight refresh missed parameters: {sorted(required-set(loaded or []))}')
    torch.cuda.synchronize()
    return {'loaded_tensors': len(loaded), 'checkpoint': str(path),
            'norm_probe': worker.model_runner.model.model.norm.weight[:8].float().tolist()}


class FullWeightWorkerExtension:
    def refresh_student_weights(self, path):
        return sync_vllm_checkpoint(self, path)
