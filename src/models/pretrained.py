"""Pretrained ImageNet weight loading for Flax ResNet-50.

Converts PyTorch torchvision ResNet-50 weights to Flax parameter tree.
Caches the converted weights as an .npz file for reuse.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp


CACHE_DIR = "data/pretrained"
CACHE_FILE = "resnet50_imagenet_flax.npz"


def _torch_to_flax_resnet50():
    """Download PyTorch ResNet-50 weights and convert to Flax param dict."""
    import torch
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    sd = model.state_dict()

    def to_np(key):
        return np.array(sd[key].cpu().numpy())

    def conv_params(prefix):
        w = to_np(f"{prefix}.weight")
        # PyTorch conv: (out, in, kH, kW) -> Flax conv: (kH, kW, in, out)
        return {"kernel": np.transpose(w, (2, 3, 1, 0))}

    def bn_params(prefix):
        return {
            "scale": to_np(f"{prefix}.weight"),
            "bias": to_np(f"{prefix}.bias"),
        }

    def bn_stats(prefix):
        return {
            "mean": to_np(f"{prefix}.running_mean"),
            "var": to_np(f"{prefix}.running_var"),
        }

    def bottleneck_params(layer_name, block_idx):
        prefix = f"{layer_name}.{block_idx}"
        p = {}
        # conv1 (1x1), bn1
        p["Conv_0"] = conv_params(f"{prefix}.conv1")
        p["BatchNorm_0"] = bn_params(f"{prefix}.bn1")
        # conv2 (3x3), bn2
        p["Conv_1"] = conv_params(f"{prefix}.conv2")
        p["BatchNorm_1"] = bn_params(f"{prefix}.bn2")
        # conv3 (1x1), bn3
        p["Conv_2"] = conv_params(f"{prefix}.conv3")
        p["BatchNorm_2"] = bn_params(f"{prefix}.bn3")
        # downsample (if exists)
        ds_key = f"{prefix}.downsample.0.weight"
        if ds_key in sd:
            p["Conv_3"] = conv_params(f"{prefix}.downsample.0")
            p["BatchNorm_3"] = bn_params(f"{prefix}.downsample.1")
        return p

    def bottleneck_stats(layer_name, block_idx):
        prefix = f"{layer_name}.{block_idx}"
        s = {}
        s["BatchNorm_0"] = bn_stats(f"{prefix}.bn1")
        s["BatchNorm_1"] = bn_stats(f"{prefix}.bn2")
        s["BatchNorm_2"] = bn_stats(f"{prefix}.bn3")
        ds_key = f"{prefix}.downsample.1.running_mean"
        if ds_key in sd:
            s["BatchNorm_3"] = bn_stats(f"{prefix}.downsample.1")
        return s

    # Build Flax param tree
    params = {}
    batch_stats = {}

    # Stem
    params["Conv_0"] = conv_params("conv1")
    params["BatchNorm_0"] = bn_params("bn1")
    batch_stats["BatchNorm_0"] = bn_stats("bn1")

    # Residual stages
    stage_names = ["layer1", "layer2", "layer3", "layer4"]
    stage_sizes = [3, 4, 6, 3]

    for stage_idx, (layer_name, n_blocks) in enumerate(
        zip(stage_names, stage_sizes)
    ):
        for block_idx in range(n_blocks):
            flax_name = f"block_{stage_idx}_{block_idx}"
            params[flax_name] = bottleneck_params(layer_name, block_idx)
            batch_stats[flax_name] = bottleneck_stats(layer_name, block_idx)

    # Note: we do NOT copy fc (classifier head) since num_classes may differ.
    # The head is randomly initialized by the Flax model.

    return params, batch_stats


def load_imagenet_resnet50(num_classes: int = 2):
    """Load pretrained ImageNet ResNet-50 weights into a Flax ResNet.

    Returns a (params, batch_stats) tuple ready to assign to a
    BatchNormTrainState. The classifier head (Dense_0) is randomly
    initialized for the target num_classes.

    Args:
        num_classes: number of output classes (head is re-initialized).
    """
    from src.models.resnet import ResNet50

    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)

    if os.path.exists(cache_path):
        data = dict(np.load(cache_path, allow_pickle=True))
        pt_params = data["params"].item()
        pt_batch_stats = data["batch_stats"].item()
    else:
        print("Converting PyTorch ResNet-50 ImageNet weights to Flax...")
        pt_params, pt_batch_stats = _torch_to_flax_resnet50()
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, params=pt_params, batch_stats=pt_batch_stats)
        print(f"Cached Flax weights to {cache_path}")

    # Initialize a fresh model to get the correct param structure
    model = ResNet50(num_classes=num_classes)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 224, 224, 3))
    variables = model.init(rng, dummy, train=True)
    full_params = variables["params"]
    full_batch_stats = variables.get("batch_stats", {})

    # Copy pretrained weights into the fresh param tree (except classifier head)
    def _merge(fresh, pretrained):
        merged = {}
        for k in fresh:
            if k in pretrained:
                if isinstance(fresh[k], dict):
                    merged[k] = _merge(fresh[k], pretrained[k])
                else:
                    merged[k] = jnp.array(pretrained[k])
            else:
                merged[k] = fresh[k]
        return merged

    params = _merge(dict(full_params), pt_params)
    batch_stats = _merge(dict(full_batch_stats), pt_batch_stats)

    return params, batch_stats
