"""
Offline tests for loading a *multimodal* Gemma 4 checkpoint (Gemma4ForConditionalGeneration).

The released checkpoints (e.g. google/gemma-4-E2B-it) are multimodal wrappers: ``model_type: gemma4``
with the language model nested under ``text_config`` and its weights stored under the
``model.language_model.*`` container prefix, alongside vision/audio towers. DRIFT serves only the text
tower, so it must (a) dispatch on the nested text config, (b) load blocks from
``model.language_model.layers.*``, and (c) strip that container prefix for the client embeddings.

Everything here is synthesized on disk and run single-process -- no download, swarm, or network.
"""
import os

import pytest
import torch
from safetensors.torch import save_file

from drift.models.gemma4.config import DistributedGemma4Config, is_multimodal_wrapper_checkpoint
from drift.models.gemma4.model import _Gemma4WrapperLoadMixin
from drift.server.from_pretrained import load_pretrained_block
from drift.utils.auto_config import AutoDistributedConfig

ATOL = 3e-5
_WRAPPER_KEY_MAPPING = {r"^model\.language_model\.": "model."}


def _tiny_text_config():
    from transformers.models.gemma4 import Gemma4TextConfig

    # Same tiny shape as test_gemma4_block: both a sliding and a full attention type have a donor
    # followed by a sharing consumer, so KV sharing is actually exercised across the stack.
    return Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        sliding_window=1024,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
        enable_moe_block=False,
    )


@pytest.fixture(scope="module")
def wrapper_checkpoint(tmp_path_factory):
    """A tiny multimodal Gemma 4 checkpoint; returns (path, the stock text model it was built from)."""
    from transformers.models.gemma4 import Gemma4Config, Gemma4TextModel

    text_cfg = _tiny_text_config()
    torch.manual_seed(0)
    text_model = Gemma4TextModel(text_cfg).eval()

    path = tmp_path_factory.mktemp("gemma4_wrapper")
    # Nest the text tower under model.language_model.* and add a throwaway vision-tower tensor that
    # must be ignored on load.
    state_dict = {f"model.language_model.{k}": v for k, v in text_model.state_dict().items()}
    state_dict["model.vision_tower.patch_embedder.position_embedding_table"] = torch.randn(4, 8)
    save_file(state_dict, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})

    wrapper_cfg = Gemma4Config(text_config=text_cfg.to_dict())
    wrapper_cfg.architectures = ["Gemma4ForConditionalGeneration"]
    wrapper_cfg.save_pretrained(path)
    return str(path), text_model


@pytest.fixture(scope="module")
def text_only_checkpoint(tmp_path_factory):
    """A tiny text-only Gemma 4 checkpoint (model_type gemma4_text, weights under model.*)."""
    from transformers.models.gemma4 import Gemma4TextModel

    text_cfg = _tiny_text_config()
    torch.manual_seed(0)
    text_model = Gemma4TextModel(text_cfg).eval()

    path = tmp_path_factory.mktemp("gemma4_text_only")
    save_file(
        {f"model.{k}": v for k, v in text_model.state_dict().items()},
        os.path.join(path, "model.safetensors"),
        metadata={"format": "pt"},
    )
    DistributedGemma4Config(**text_cfg.to_dict()).save_pretrained(path)
    return str(path)


def test_wrapper_detection(wrapper_checkpoint, text_only_checkpoint):
    path, _ = wrapper_checkpoint
    assert is_multimodal_wrapper_checkpoint(path) is True
    assert is_multimodal_wrapper_checkpoint(text_only_checkpoint) is False


def test_config_dispatch_uses_nested_text_config(wrapper_checkpoint):
    path, _ = wrapper_checkpoint
    cfg = AutoDistributedConfig.from_pretrained(path)
    assert type(cfg).__name__ == "DistributedGemma4Config"
    assert cfg.model_type == "gemma4_text"  # dispatched onto the nested text config
    assert cfg.num_hidden_layers == 6 and cfg.num_kv_shared_layers == 2
    assert cfg.block_prefix == "model.language_model.layers"


def test_text_only_checkpoint_keeps_default_block_prefix(text_only_checkpoint):
    cfg = AutoDistributedConfig.from_pretrained(text_only_checkpoint)
    assert cfg.block_prefix == "model.layers"


def test_wrapper_blocks_load_bit_exact(wrapper_checkpoint):
    """Every block loads from model.language_model.layers.* and matches the stock layer output,
    including the donor/consumer KV-sharing layers."""
    path, text_model = wrapper_checkpoint
    cfg = AutoDistributedConfig.from_pretrained(path)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    layer_in, layer_out = {}, {}

    def pre_hook(i):
        def hook(_m, args, kwargs):
            layer_in[i] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

        return hook

    def out_hook(i):
        def hook(_m, _args, _kwargs, output):
            layer_out[i] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        return hook

    for i, layer in enumerate(text_model.layers):
        layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True)
        layer.register_forward_hook(out_hook(i), with_kwargs=True)
    with torch.inference_mode():
        text_model(input_ids, use_cache=False)
        embeds = text_model.embed_tokens(input_ids)
        per_layer_inputs = text_model.get_per_layer_inputs(input_ids, embeds)
        per_layer_inputs = text_model.project_per_layer_inputs(embeds, per_layer_inputs)

    shared_kv_states = {}
    for i in range(cfg.num_hidden_layers):
        block = load_pretrained_block(path, i, config=cfg, torch_dtype=torch.float32).eval()
        with torch.inference_mode():
            (out,) = block(layer_in[i], per_layer_input=per_layer_inputs[:, :, i, :], shared_kv_states=shared_kv_states)
        assert torch.allclose(out, layer_out[i], atol=ATOL), (i, (out - layer_out[i]).abs().max().item())

    assert set(shared_kv_states) == {"sliding_attention", "full_attention"}


def test_wrapper_mixin_injects_key_mapping(wrapper_checkpoint, text_only_checkpoint):
    """The load mixin injects the language_model-stripping key_mapping only for wrapper checkpoints."""
    path, _ = wrapper_checkpoint
    captured = {}

    class _Base:
        @classmethod
        def from_pretrained(cls, model_name_or_path, *args, **kwargs):
            captured["kwargs"] = kwargs
            return "loaded"

    class _Model(_Gemma4WrapperLoadMixin, _Base):
        pass

    assert _Model.from_pretrained(path) == "loaded"
    assert captured["kwargs"].get("key_mapping") == _WRAPPER_KEY_MAPPING

    captured.clear()
    _Model.from_pretrained(text_only_checkpoint)
    assert "key_mapping" not in captured["kwargs"]


def test_wrapper_key_mapping_loads_text_tower(wrapper_checkpoint):
    """The injected key_mapping lands the text tower's weights correctly.

    Loaded into a stock text ForCausalLM (whose .model mirrors the distributed client's, minus the
    remote layers) so the check needs no DHT.
    """
    from transformers.models.gemma4 import Gemma4ForCausalLM

    path, text_model = wrapper_checkpoint
    ref = Gemma4ForCausalLM.from_pretrained(path, key_mapping=_WRAPPER_KEY_MAPPING, torch_dtype=torch.float32)
    assert torch.equal(ref.model.embed_tokens.weight, text_model.embed_tokens.weight)
    assert torch.equal(ref.model.norm.weight, text_model.norm.weight)
    assert torch.equal(ref.model.per_layer_model_projection.weight, text_model.per_layer_model_projection.weight)
    assert torch.equal(ref.model.embed_tokens_per_layer.weight, text_model.embed_tokens_per_layer.weight)
