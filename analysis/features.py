"""
Feature extraction utilities using PyTorch forward hooks.

Collects spatially-pooled features [N, C] from named layers during
a forward pass, for downstream CKA and drift analysis.
"""

import torch
import torch.nn.functional as F


def _get_module_by_name(model, name):
    """Resolve a dot-separated module name to the actual nn.Module."""
    cur = model
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


class FeatureGrabber:
    """
    Attaches forward hooks to named layers and caches their
    spatially-pooled outputs as [B, C] tensors on CPU.
    """

    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.cache = {k: [] for k in layer_names}
        self.handles = []

        for lname in layer_names:
            mod = _get_module_by_name(model, lname)
            h = mod.register_forward_hook(self._make_hook(lname))
            self.handles.append(h)

    def _make_hook(self, lname):
        def hook(module, inp, out):
            x = out
            if x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            elif x.dim() != 2:
                x = x.view(x.size(0), -1)
            self.cache[lname].append(x.detach().float().cpu())
        return hook

    def clear(self):
        for k in self.cache:
            self.cache[k] = []

    def close(self):
        for h in self.handles:
            h.remove()


def collect_features(model, loader, layer_names, max_images=256, device="cuda"):
    """
    Run *model* on up to *max_images* from *loader* and return
    ``{layer_name: Tensor[N, C]}`` of pooled intermediate features.
    """
    model.eval()
    grab = FeatureGrabber(model, layer_names)
    grab.clear()

    n = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            model(xb)
            n += xb.size(0)
            if n >= max_images:
                break

    feats = {}
    for lname in layer_names:
        feats[lname] = torch.cat(grab.cache[lname], dim=0)[:max_images]
    grab.close()
    return feats


def strip_prefix(feats_dict, prefix="m."):
    """Rename keys by stripping a prefix (e.g. 'm.prep' → 'prep')."""
    return {k.replace(prefix, "", 1): v for k, v in feats_dict.items()}
