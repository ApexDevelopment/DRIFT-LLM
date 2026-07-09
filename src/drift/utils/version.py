import os
import re
from typing import Union

from hivemind.utils.logging import TextStyle, get_logger

import drift

logger = get_logger(__name__)


def log_version() -> None:
    logger.info(f"Running {TextStyle.BOLD}DRIFT-LLM {drift.__version__}{TextStyle.RESET}")


def get_compatible_model_repo(model_name_or_path: Union[str, os.PathLike, None]) -> Union[str, os.PathLike, None]:
    if model_name_or_path is None:
        return None

    match = re.fullmatch(r"(bigscience/.+)-drift", str(model_name_or_path))
    if match is None:
        return model_name_or_path

    logger.info(
        f"Loading model from {match.group(1)}, since DRIFT-LLM 1.2.0+ uses original repos instead of converted ones"
    )
    return match.group(1)
