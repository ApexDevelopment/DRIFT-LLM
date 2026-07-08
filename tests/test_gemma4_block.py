"""
Offline exact-match tests for the Gemma 4 block (per-layer inputs + KV sharing).

Checks the wrapped block against a tiny stock HF ``Gemma4TextModel`` across all four layer roles
(sliding/full x donor/sharing), the per-layer-input side channel, and the BLOOM cache round-trip on
a non-sharing layer. No swarm / network / download required (does not import test_utils).
"""
import pytest
import torch

from drift.models.gemma4.block import WrappedGemma4Block

ATOL = 3e-5


def _tiny_config(impl="eager"):
    from transformers.models.gemma4 import Gemma4TextConfig

    # 6 layers, types chosen so both a sliding and a full attention type have a non-shared donor
    # (last non-shared of that type) followed by a sharing consumer:
    #   idx:        0        1        2      3        4        5
    #   type:    sliding  sliding   full  sliding  sliding   full
    #   shared:     -        -       -      -        yes      yes   (num_kv_shared_layers=2)
    #   donor:                     full  sliding
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,  # multi-query attention, as in the released checkpoints
        head_dim=16,
        global_head_dim=32,  # full-attention layers use a wider head than sliding ones
        layer_types=layer_types,
        num_kv_shared_layers=2,
        sliding_window=1024,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
        enable_moe_block=False,
    )
    cfg._attn_implementation = impl
    return cfg


def _model_and_per_layer_inputs(cfg, input_ids):
    """Build a stock model, capture each layer's exact input/output, and the per-layer inputs."""
    from transformers.models.gemma4 import Gemma4TextModel

    torch.manual_seed(0)
    model = Gemma4TextModel(cfg).eval()

    layer_in, layer_out = {}, {}

    def pre_hook(idx):
        def hook(_module, args, kwargs):
            layer_in[idx] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

        return hook

    def out_hook(idx):
        def hook(_module, _args, _kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            layer_out[idx] = out.detach().clone()

        return hook

    for i, layer in enumerate(model.layers):
        layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True)
        layer.register_forward_hook(out_hook(i), with_kwargs=True)

    with torch.inference_mode():
        model(input_ids, use_cache=False)
        inputs_embeds = model.embed_tokens(input_ids)
        per_layer_inputs = model.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = model.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

    return model, layer_in, layer_out, per_layer_inputs


def test_gemma4_block_stack_matches_hf():
    cfg = _tiny_config()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    model, layer_in, layer_out, per_layer_inputs = _model_and_per_layer_inputs(cfg, input_ids)

    shared_kv_states = {}  # threaded through the stack exactly as the stock model does
    for i in range(cfg.num_hidden_layers):
        block = WrappedGemma4Block(cfg, layer_idx=i).eval()
        missing, unexpected = block.load_state_dict(model.layers[i].state_dict(), strict=False)
        assert not unexpected, f"layer {i} unexpected keys: {unexpected}"
        assert all("rotary" in name for name in missing), f"layer {i} unexpected missing keys: {missing}"

        with torch.inference_mode():
            (out,) = block(
                layer_in[i],
                per_layer_input=per_layer_inputs[:, :, i, :],
                shared_kv_states=shared_kv_states,
            )
        assert torch.allclose(out, layer_out[i], atol=ATOL), (
            f"layer {i} ({cfg.layer_types[i]}, shared={block.is_kv_shared_layer}): "
            f"{(out - layer_out[i]).abs().max().item()}"
        )

    # The stack must have actually exercised KV sharing (both donors fired, both consumers read).
    assert set(shared_kv_states) == {"sliding_attention", "full_attention"}


def test_gemma4_nonshared_block_cache_roundtrip():
    cfg = _tiny_config()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    model, layer_in, layer_out, per_layer_inputs = _model_and_per_layer_inputs(cfg, input_ids)

    layer_idx = 0  # non-sharing, non-donor sliding layer -> needs no shared_kv_states
    block = WrappedGemma4Block(cfg, layer_idx=layer_idx).eval()
    block.load_state_dict(model.layers[layer_idx].state_dict(), strict=False)

    x, pli = layer_in[layer_idx], per_layer_inputs[:, :, layer_idx, :]
    n = input_ids.shape[1]
    with torch.inference_mode():
        prefill_out, _ = block(x, per_layer_input=pli, use_cache=True)

        past, steps = None, []
        for i in range(n):
            step_out, past = block(x[:, i : i + 1], per_layer_input=pli[:, i : i + 1], layer_past=past, use_cache=True)
            steps.append(step_out)
        stepwise_out = torch.cat(steps, dim=1)

    assert torch.allclose(prefill_out, layer_out[layer_idx], atol=ATOL), (
        (prefill_out - layer_out[layer_idx]).abs().max()
    )
    assert torch.allclose(stepwise_out, layer_out[layer_idx], atol=ATOL), (
        (stepwise_out - layer_out[layer_idx]).abs().max()
    )
