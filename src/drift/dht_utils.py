import warnings

warnings.warn(
    "drift.dht_utils has been moved to drift.utils.dht. This alias will be removed in DRIFT-LLM 2.2.0+",
    DeprecationWarning,
    stacklevel=2,
)

from drift.utils.dht import *
