"""
Offline exact-match tests for the DeepSeek-V3 block (Multi-head Latent Attention + MoE).

Checks a dense layer and a MoE layer against a tiny stock HF model, the asymmetric-head-dim
(key != value) BLOOM cache round-trip, and MLACache descriptor sizing. No swarm / network /
download required (does not import test_utils).
"""
import pytest
import torch

from petals.models.deepseek_v3.block import WrappedDeepseekV3Block
from petals.utils.kv_cache import MLACache

ATOL = 3e-5


def _tiny_config(impl="eager"):
    from transformers.models.deepseek_v3 import DeepseekV3Config

    cfg = DeepseekV3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
    )
    cfg._attn_implementation = impl
    return cfg


@pytest.mark.parametrize("layer_idx", [0, 1])  # 0 = dense FFN, 1 = MoE (first_k_dense_replace=1)
def test_deepseek_v3_block_matches_hf(layer_idx):
    from transformers.models.deepseek_v3 import DeepseekV3Model

    torch.manual_seed(0)
    cfg = _tiny_config()
    model = DeepseekV3Model(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.inference_mode():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    layer_in, layer_ref = out.hidden_states[layer_idx], out.hidden_states[layer_idx + 1]

    block = WrappedDeepseekV3Block(cfg, layer_idx=layer_idx).eval()
    missing, unexpected = block.load_state_dict(model.layers[layer_idx].state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("rotary" in name for name in missing), f"unexpected missing keys: {missing}"

    n = input_ids.shape[1]
    with torch.inference_mode():
        prefill_out, full_kv = block(layer_in, use_cache=True)
        past, steps = None, []
        for i in range(n):
            step_out, past = block(layer_in[:, i : i + 1], layer_past=past, use_cache=True)
            steps.append(step_out)
        stepwise_out = torch.cat(steps, dim=1)

    assert torch.allclose(prefill_out, layer_ref, atol=ATOL), (prefill_out - layer_ref).abs().max()
    assert torch.allclose(stepwise_out, layer_ref, atol=ATOL), (stepwise_out - layer_ref).abs().max()
    # MLA keys and values have different head dims; the round-trip must preserve both
    assert full_kv[0].shape[1] == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim  # key head dim (BLOOM: [b*kv, hd, seq])
    assert full_kv[1].shape[2] == cfg.v_head_dim  # value head dim
    assert torch.allclose(full_kv[0], past[0], atol=ATOL)
    assert torch.allclose(full_kv[1], past[1], atol=ATOL)


def test_mla_cache_descriptor_asymmetric_head_dims():
    cfg = _tiny_config()
    cfg.num_key_value_groups = cfg.num_attention_heads // cfg.num_key_value_heads
    keys, values = MLACache(cfg).get_cache_descriptors(
        1, 16, dtype=torch.float32, devices=[torch.device("cpu")], shard_num_heads=[cfg.num_attention_heads]
    )
    assert keys.shape == (1, cfg.num_key_value_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim, 16)
    assert values.shape == (1, cfg.num_key_value_heads, 16, cfg.v_head_dim)
