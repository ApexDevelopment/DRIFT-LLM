import pytest

from drift import AutoDistributedModelForCausalLM


def _save_tiny_model(tmp_path, config_cls, model_cls):
    config = config_cls(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    model_cls(config).save_pretrained(tmp_path, safe_serialization=True)


@pytest.mark.parametrize(
    ("config_path", "model_path"),
    [
        ("transformers.models.qwen2.Qwen2Config", "transformers.models.qwen2.Qwen2ForCausalLM"),
        ("transformers.models.qwen3.Qwen3Config", "transformers.models.qwen3.Qwen3ForCausalLM"),
    ],
)
def test_tied_embeddings_load_with_ignored_remote_blocks(tmp_path, config_path, model_path):
    config_module, config_name = config_path.rsplit(".", 1)
    model_module, model_name = model_path.rsplit(".", 1)

    config_cls = getattr(__import__(config_module, fromlist=[config_name]), config_name)
    model_cls = getattr(__import__(model_module, fromlist=[model_name]), model_name)

    _save_tiny_model(tmp_path, config_cls, model_cls)

    model = AutoDistributedModelForCausalLM.from_pretrained(
        tmp_path,
        dht_prefix="test",
        low_cpu_mem_usage=True,
    )

    assert model.lm_head.weight is model.model.embed_tokens.weight
