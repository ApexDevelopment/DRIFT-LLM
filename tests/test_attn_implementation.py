"""
Offline tests for server-side attention-implementation selection
(``petals.utils.misc.default_attn_implementation`` + the per-block wiring).

sdpa is the fast default (torch SDPA -> FlashAttention/memory-efficient kernels on GPU) wherever
it is numerically correct; ALiBi (Bloom/Falcon) and Gemma-2 logit softcapping must stay on eager.
No swarm / network / download required.
"""
import pytest
import torch

from petals.utils.misc import default_attn_implementation


def test_default_attn_implementation_per_arch():
    from transformers.models.bloom import BloomConfig
    from transformers.models.falcon import FalconConfig
    from transformers.models.gemma2 import Gemma2Config
    from transformers.models.gemma3 import Gemma3TextConfig
    from transformers.models.llama import LlamaConfig
    from transformers.models.mixtral import MixtralConfig
    from transformers.models.qwen2 import Qwen2Config

    assert default_attn_implementation(LlamaConfig()) == "sdpa"
    assert default_attn_implementation(Qwen2Config()) == "sdpa"
    assert default_attn_implementation(MixtralConfig()) == "sdpa"
    assert default_attn_implementation(Gemma3TextConfig()) == "sdpa"
    # ALiBi and logit softcapping force eager:
    assert default_attn_implementation(BloomConfig()) == "eager"
    assert default_attn_implementation(FalconConfig(alibi=True)) == "eager"
    assert default_attn_implementation(FalconConfig(alibi=False, new_decoder_architecture=True)) == "sdpa"
    assert default_attn_implementation(Gemma2Config()) == "eager"  # attn_logit_softcapping


def _make(arch):
    if arch == "llama":
        from transformers.models.llama import LlamaConfig

        from petals.models.llama.block import WrappedLlamaBlock

        return LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                           intermediate_size=128, num_hidden_layers=2, vocab_size=128), WrappedLlamaBlock
    if arch == "qwen3":
        from transformers.models.qwen3 import Qwen3Config

        from petals.models.qwen3.block import WrappedQwen3Block

        return Qwen3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=48,
                           intermediate_size=128, num_hidden_layers=2, vocab_size=128), WrappedQwen3Block
    if arch == "gemma3":
        from transformers.models.gemma3 import Gemma3TextConfig

        from petals.models.gemma3.block import WrappedGemma3Block

        return Gemma3TextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                intermediate_size=128, num_hidden_layers=2, vocab_size=128,
                                sliding_window=8, sliding_window_pattern=6, query_pre_attn_scalar=64), WrappedGemma3Block
    raise ValueError(arch)


@pytest.mark.parametrize("arch", ["llama", "qwen3", "gemma3"])
def test_sdpa_matches_eager(arch):
    """For the sdpa-safe archs, swapping eager -> sdpa must not change results (fp32, CPU)."""
    def build(impl):
        torch.manual_seed(0)
        cfg, BlockCls = _make(arch)
        cfg._attn_implementation = impl
        return BlockCls(cfg).eval()

    eager, sdpa = build("eager"), build("sdpa")
    sdpa.load_state_dict(eager.state_dict(), strict=False)
    x = torch.randn(1, 6, eager.config.hidden_size)
    with torch.inference_mode():
        out_eager, _ = eager(x, use_cache=True)
        out_sdpa, _ = sdpa(x, use_cache=True)
    assert torch.allclose(out_eager, out_sdpa, atol=1e-4), (out_eager - out_sdpa).abs().max()
