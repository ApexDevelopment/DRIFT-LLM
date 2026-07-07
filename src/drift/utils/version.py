import os
import re
from typing import Union

import requests
from hivemind.utils.logging import TextStyle, get_logger
from packaging.version import parse

import drift

logger = get_logger(__name__)


def validate_version() -> None:
    logger.info(f"Running {TextStyle.BOLD}DRIFT-LLM {drift.__version__}{TextStyle.RESET}")
    try:
        r = requests.get("https://pypi.python.org/pypi/drift/json")
        r.raise_for_status()
        response = r.json()

        versions = [parse(ver) for ver in response.get("releases")]
        latest = max(ver for ver in versions if not ver.is_prerelease)

        if parse(drift.__version__) < latest:
            logger.info(
                f"A newer version {latest} is available. Please upgrade with: "
                f"{TextStyle.BOLD}pip install --upgrade drift{TextStyle.RESET}"
            )
    except Exception as e:
        logger.warning("Failed to fetch the latest DRIFT-LLM version from PyPI:", exc_info=True)


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
