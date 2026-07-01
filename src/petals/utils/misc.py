import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters

DUMMY_INT64 = torch.empty(0, dtype=torch.int64)

DUMMY_KEY_PAST = torch.empty((0, 0, 0))


def is_dummy(tensor: torch.Tensor) -> bool:
    return tensor.numel() == 0


SPECIAL_DTYPE_SIZES = {torch.bool: 1, torch.qint8: 1, torch.qint32: 4}


def get_size_in_bytes(dtype: torch.dtype) -> int:
    if dtype in SPECIAL_DTYPE_SIZES:
        return SPECIAL_DTYPE_SIZES[dtype]
    get_info = torch.finfo if dtype.is_floating_point else torch.iinfo
    return (get_info(dtype).bits * (1 + dtype.is_complex)) // 8


def get_num_attention_heads(attn_module: torch.nn.Module, config) -> int:
    """Number of query heads held by an attention module.

    transformers >=5.0 removed the ``num_heads`` attribute from several attention classes
    (e.g. ``LlamaAttention``), so fall back to deriving it from the query projection. Older
    attention modules that still expose ``num_heads`` (used to count heads per tensor-parallel
    shard) keep working unchanged.
    """
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is not None:
        return num_heads
    head_dim = getattr(attn_module, "head_dim", config.hidden_size // config.num_attention_heads)
    for proj_name in ("q_proj", "query_key_value", "query"):
        proj = getattr(attn_module, proj_name, None)
        if proj is not None:
            if proj_name == "query_key_value":
                # fused QKV projection (e.g. BLOOM/Falcon): query heads = total heads
                return config.num_attention_heads
            # tensor_parallel replaces sliced Linears with a wrapper exposing the shard via `.module`
            if not hasattr(proj, "out_features") and hasattr(proj, "module"):
                proj = proj.module
            return proj.out_features // head_dim
    return config.num_attention_heads


def docstring_from(source):
    def add_docstring(dest):
        dest.__doc__ = source.__doc__
        return dest

    return add_docstring
