"""
Offline exact-match tests for the dense (MHA/GQA) block wrappers.

These build a tiny randomly-initialized HF model on CPU and check that Petals' block wrapper
reproduces the stock decoder layer bit-for-bit at prefill, plus a stepwise round-trip through
Petals' BLOOM-layout KV cache. They need no swarm, no network and no model download, so unlike
``test_block_exact_match.py`` they do not import ``test_utils`` / require ``INITIAL_PEERS``.
"""
import pytest
import torch

# A sequence longer than the sliding window below, so sliding vs full attention actually diverge.
SEQ_LEN = 10
SLIDING_WINDOW = 8
ATOL = 2e-5


def _make(arch):
    """Return (config, HF model class, Petals block class) with a tiny config for ``arch``."""
    if arch == "llama":
        from transformers.models.llama import LlamaConfig, LlamaModel

        from petals.models.llama.block import WrappedLlamaBlock

        cfg = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=128,
        )
        return cfg, LlamaModel, WrappedLlamaBlock
    if arch == "mistral":
        from transformers.models.mistral import MistralConfig, MistralModel

        from petals.models.mistral.block import WrappedMistralBlock

        cfg = MistralConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=128,
            sliding_window=SLIDING_WINDOW,
        )
        return cfg, MistralModel, WrappedMistralBlock
    if arch == "qwen2":
        from transformers.models.qwen2 import Qwen2Config, Qwen2Model

        from petals.models.qwen2.block import WrappedQwen2Block

        cfg = Qwen2Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=128,
        )
        return cfg, Qwen2Model, WrappedQwen2Block
    if arch == "qwen3":
        from transformers.models.qwen3 import Qwen3Config, Qwen3Model

        from petals.models.qwen3.block import WrappedQwen3Block

        cfg = Qwen3Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=48,
            intermediate_size=128,
            num_hidden_layers=2,
            vocab_size=128,
        )
        return cfg, Qwen3Model, WrappedQwen3Block
    if arch == "gemma2":
        from transformers.models.gemma2 import Gemma2Config, Gemma2Model

        from petals.models.gemma2.block import WrappedGemma2Block

        cfg = Gemma2Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            num_hidden_layers=4,
            vocab_size=128,
            sliding_window=SLIDING_WINDOW,
            query_pre_attn_scalar=16,
        )
        return cfg, Gemma2Model, WrappedGemma2Block
    if arch == "gemma3":
        from transformers.models.gemma3 import Gemma3TextConfig, Gemma3TextModel

        from petals.models.gemma3.block import WrappedGemma3Block

        cfg = Gemma3TextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            num_hidden_layers=7,
            vocab_size=128,
            sliding_window=SLIDING_WINDOW,
            sliding_window_pattern=6,
            query_pre_attn_scalar=64,
        )
        return cfg, Gemma3TextModel, WrappedGemma3Block
    raise ValueError(arch)


def _check_block(arch, layer_idx):
    torch.manual_seed(0)
    cfg, ModelCls, BlockCls = _make(arch)
    cfg._attn_implementation = "eager"
    model = ModelCls(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN))
    with torch.inference_mode():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    layer_in = out.hidden_states[layer_idx]  # input to layer `layer_idx`
    layer_ref = out.hidden_states[layer_idx + 1]  # its output (not the final entry, so no final norm)

    block = BlockCls(cfg, layer_idx=layer_idx).eval()
    missing, unexpected = block.load_state_dict(model.layers[layer_idx].state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("rotary" in name for name in missing), f"unexpected missing keys: {missing}"

    with torch.inference_mode():
        prefill_out, full_kv = block(layer_in, use_cache=True)
        past, steps = None, []
        for i in range(SEQ_LEN):
            step_out, past = block(layer_in[:, i : i + 1], layer_past=past, use_cache=True)
            steps.append(step_out)
        stepwise_out = torch.cat(steps, dim=1)

    assert torch.allclose(prefill_out, layer_ref, atol=ATOL), (prefill_out - layer_ref).abs().max()
    assert torch.allclose(stepwise_out, layer_ref, atol=ATOL), (stepwise_out - layer_ref).abs().max()
    # the stepwise cache matches the prefill cache after a full BLOOM-layout round-trip
    assert torch.allclose(full_kv[0], past[0], atol=ATOL)
    assert torch.allclose(full_kv[1], past[1], atol=ATOL)


@pytest.mark.parametrize("arch", ["llama", "mistral", "qwen2", "qwen3"])
def test_dense_block_matches_hf(arch):
    _check_block(arch, layer_idx=0)


@pytest.mark.parametrize("arch", ["gemma2", "gemma3"])
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_gemma_block_matches_hf(arch, layer_type):
    cfg, _, _ = _make(arch)
    last = cfg.num_hidden_layers - 1
    # a non-last layer of the requested type (last entry of hidden_states carries the final norm)
    layer_idx = next(i for i, t in enumerate(cfg.layer_types) if t == layer_type and i != last)
    _check_block(arch, layer_idx=layer_idx)
