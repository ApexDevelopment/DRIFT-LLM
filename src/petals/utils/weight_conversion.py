"""
Apply transformers' checkpoint weight-conversion mapping to a single block's state dict.

Since transformers 5.0, loading a pretrained model can *restructure* weights (e.g. Mixtral
fuses per-expert ``block_sparse_moe.experts.N.{w1,w2,w3}`` into ``mlp.experts.{gate_up_proj,
down_proj}``). transformers does this inside its own loading pipeline, which Petals bypasses
(it matches checkpoint tensor names directly to a single block's parameters to avoid
instantiating the whole model).

This module reuses transformers' own conversion mapping and ``ConversionOps`` so that any
architecture transformers can convert, Petals can too — without hard-coding per-model fusions.
"""
import re
from functools import lru_cache
from typing import Dict, List

import torch
from accelerate import init_empty_weights
from hivemind.utils.logging import get_logger
from transformers import AutoConfig, AutoModel

logger = get_logger(__name__)

# A synthetic prefix so that mapping patterns like ".block_sparse_moe." / ".experts.*.w1.weight"
# (which assume a per-layer context) match block-relative checkpoint keys.
_LAYER_PREFIX = "model.layers.0."


@lru_cache(maxsize=None)
def _get_weight_mapping(model_type: str):
    """Return transformers' weight-conversion mapping for a model_type, or () if none.

    The mapping is architectural (which modules get fused/renamed), so a minimal default base
    model is enough. Built once per model_type from a cheap meta-instantiated model.
    """
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
    except ImportError:
        return ()

    try:
        base_config = AutoConfig.for_model(model_type)
        base_config.num_hidden_layers = 1  # keep the meta model tiny; the mapping is layer-agnostic
        with init_empty_weights():
            model = AutoModel.from_config(base_config)
        return tuple(get_model_conversion_mapping(model) or [])
    except Exception as e:
        logger.warning(f"Could not build weight-conversion mapping for model_type={model_type!r}: {e}")
        return ()


def _needs_conversion(state_dict: Dict[str, torch.Tensor], block: torch.nn.Module) -> bool:
    """True if some block parameter is missing from the checkpoint keys (i.e. names differ)."""
    return any(name not in state_dict for name, _ in block.named_parameters())


def _suffix_regex(pattern: str) -> re.Pattern:
    # source patterns are suffix patterns (e.g. ".experts.*.w1.weight"); capture the prefix
    # before them and the wildcard index (the module-list / expert number)
    return re.compile("^(?P<prefix>.*)" + re.escape(pattern).replace(r"\*", r"(?P<idx>\d+)") + "$")


def maybe_convert_block_state_dict(
    config, state_dict: Dict[str, torch.Tensor], block: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """Apply transformers' conversion mapping to a block-relative ``state_dict`` if needed.

    Returns the state dict unchanged when the checkpoint already matches the block's parameter
    names (the common case, e.g. Llama), so this is a no-op for architectures with no rewrites.
    """
    if not _needs_conversion(state_dict, block):
        return state_dict

    mapping = _get_weight_mapping(config.model_type)
    if not mapping:
        return state_dict

    from transformers.core_model_loading import WeightConverter, WeightRenaming

    # Work on prefixed keys so per-layer rename patterns (e.g. ".block_sparse_moe.") match, then
    # strip the prefix at the end.
    sd = {_LAYER_PREFIX + k: v for k, v in state_dict.items()}

    for transform in mapping:
        if isinstance(transform, WeightRenaming):
            src, tgt = transform.source_patterns[0], transform.target_patterns[0]
            sd = {key.replace(src, tgt): value for key, value in sd.items()}
        elif isinstance(transform, WeightConverter):
            sd = _apply_converter(sd, transform)

    return {key[len(_LAYER_PREFIX) :] if key.startswith(_LAYER_PREFIX) else key: value for key, value in sd.items()}


def _apply_converter(sd: Dict[str, torch.Tensor], converter) -> Dict[str, torch.Tensor]:
    """Collect the tensors matching a WeightConverter's source patterns and run its ops.

    Grouped by the key prefix before the (suffix) source pattern, so multiple modules/layers are
    handled independently. For Petals' single block there is exactly one group.
    """
    # prefix -> {source_pattern -> [(idx, key)]}
    groups: Dict[str, Dict[str, List]] = {}
    for source_pattern in converter.source_patterns:
        regex = _suffix_regex(source_pattern)
        for key in sd:
            m = regex.match(key)
            if m:
                prefix = m.group("prefix")
                idx = int(m.group("idx")) if "idx" in m.groupdict() and m.group("idx") is not None else 0
                groups.setdefault(prefix, {}).setdefault(source_pattern, []).append((idx, key))

    if not groups:
        return sd

    out = dict(sd)
    for prefix, per_pattern in groups.items():
        if set(per_pattern) != set(converter.source_patterns):
            continue  # incomplete match for this prefix; leave untouched
        input_dict = {}
        consumed = []
        for source_pattern in converter.source_patterns:
            matched = sorted(per_pattern[source_pattern])
            input_dict[source_pattern] = [sd[key] for _, key in matched]
            consumed.extend(key for _, key in matched)

        result = input_dict
        for op in converter.operations:
            result = op.convert(result, list(converter.source_patterns), converter.target_patterns)

        for key in consumed:
            out.pop(key, None)
        for target_pattern, tensor in result.items():
            out[prefix + target_pattern] = tensor
    return out
