"""
Offline tests for the pluggable KV-cache strategy (``drift.utils.kv_cache``).

Covers the two things ``TransformerBackend`` delegates to a strategy: correct cache-descriptor
sizing (head_dim / per-shard KV heads) and a prefill+decode round-trip that mirrors what
``TransformerBackend.inference_step`` does. No swarm / network / model download required.
"""
import pytest
import torch

from drift.utils.kv_cache import StandardGQACache

ATOL = 3e-5


def _llama_config(head_dim=None):
    from transformers.models.llama import LlamaConfig

    hidden, heads, kv = 64, 4, 2
    kwargs = dict(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=128,
    )
    if head_dim is not None:
        kwargs["head_dim"] = head_dim
    cfg = LlamaConfig(**kwargs)
    cfg.num_key_value_groups = heads // kv
    return cfg


def test_descriptor_uses_explicit_head_dim():
    # An explicit config.head_dim (Gemma / some Qwen3) must win over hidden_size // num_heads.
    cfg = _llama_config(head_dim=48)  # 48 != 64 // 4 == 16
    strategy = StandardGQACache(cfg)
    keys, values = strategy.get_cache_descriptors(
        1, 32, dtype=torch.float32, devices=[torch.device("cpu")], shard_num_heads=[cfg.num_attention_heads]
    )
    assert keys.shape == (1, cfg.num_key_value_heads, 48, 32)
    assert values.shape == (1, cfg.num_key_value_heads, 32, 48)


def test_descriptor_per_shard_kv_heads():
    # Two tensor-parallel shards each hold shard_query_heads // num_key_value_groups KV heads,
    # not the full num_key_value_heads (the bug this replaced).
    cfg = _llama_config()  # 4 query heads, 2 KV heads, groups=2, head_dim=16
    strategy = StandardGQACache(cfg)
    devices = [torch.device("cpu"), torch.device("cpu")]
    descriptors = strategy.get_cache_descriptors(1, 32, dtype=torch.float32, devices=devices, shard_num_heads=[2, 2])
    assert len(descriptors) == 4  # (key, value) per shard
    for keys, values in (descriptors[0:2], descriptors[2:4]):
        assert keys.shape == (1, 1, 16, 32)  # 2 query heads // groups(2) == 1 KV head per shard
        assert values.shape == (1, 1, 32, 16)


@pytest.mark.parametrize("head_dim", [None, 48])
def test_standard_gqa_cache_roundtrip(head_dim):
    from drift.models.llama.block import WrappedLlamaBlock

    torch.manual_seed(0)
    cfg = _llama_config(head_dim=head_dim)
    cfg._attn_implementation = "eager"
    block = WrappedLlamaBlock(cfg).eval()
    strategy = StandardGQACache(cfg)
    devices, shard_num_heads = [torch.device("cpu")], [cfg.num_attention_heads]

    b, n_prefill, total, hid = 1, 5, 8, cfg.hidden_size
    x = torch.randn(b, total, hid)
    with torch.inference_mode():
        ref_out, _ = block(x, use_cache=True)

    descriptors = strategy.get_cache_descriptors(
        b, 32, dtype=torch.float32, devices=devices, shard_num_heads=shard_num_heads
    )
    cache = [torch.zeros(*d.shape, dtype=d.dtype) for d in descriptors]

    outs = []
    with torch.inference_mode():
        past = strategy.select_layer_past(cache, 0, num_shards=1)  # empty prefix
        out, new_kvs = block(x[:, :n_prefill], layer_past=past, use_cache=True)
        strategy.update_cache(cache, new_kvs, 0)
        outs.append(out)
        for t in range(n_prefill, total):  # decode one token at a time
            past = strategy.select_layer_past(cache, t, num_shards=1)
            out, new_kvs = block(x[:, t : t + 1], layer_past=past, use_cache=True)
            strategy.update_cache(cache, new_kvs, t)
            outs.append(out)
    got = torch.cat(outs, dim=1)

    assert torch.allclose(got, ref_out, atol=ATOL), (got - ref_out).abs().max()
