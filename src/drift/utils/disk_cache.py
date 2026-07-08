import os
import shutil
from pathlib import Path
from typing import Optional, Union

import huggingface_hub
from hivemind.utils.logging import get_logger
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

from drift.utils.file_lock import file_lock

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = os.getenv("DRIFT_CACHE", Path(Path.home(), ".cache", "drift"))

BLOCKS_LOCK_FILE = "blocks.lock"


def get_file_from_repo(
    repo_id: str,
    filename: str,
    *,
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
) -> Optional[str]:
    """Drop-in replacement for ``transformers.utils.get_file_from_repo`` (removed in transformers 5.x).

    Returns the local path to the (cached or freshly downloaded) file, or ``None`` if the file
    does not exist in the repo, or is not cached when ``local_files_only=True``.

    Also supports serving a model straight from a local directory: if ``repo_id`` is a directory,
    the file is looked up inside it (returning ``None`` when absent) instead of hitting the Hub.
    """
    if os.path.isdir(repo_id):
        candidate = os.path.join(repo_id, filename)
        return candidate if os.path.isfile(candidate) else None

    try:
        return huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None


def allow_cache_reads(cache_dir: Optional[str]):
    """Allows simultaneous reads, guarantees that blocks won't be removed along the way (shared lock)"""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    return file_lock(Path(cache_dir, BLOCKS_LOCK_FILE), exclusive=False)


def allow_cache_writes(cache_dir: Optional[str]):
    """Allows saving new blocks and removing the old ones (exclusive lock)"""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    return file_lock(Path(cache_dir, BLOCKS_LOCK_FILE), exclusive=True)


def free_disk_space_for(
    size: int,
    *,
    cache_dir: Optional[str],
    max_disk_space: Optional[int],
    os_quota: int = 1024**3,  # Minimal space we should leave to keep OS function normally
):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_info = huggingface_hub.scan_cache_dir(cache_dir)

    available_space = shutil.disk_usage(cache_dir).free - os_quota
    if max_disk_space is not None:
        available_space = min(available_space, max_disk_space - cache_info.size_on_disk)

    gib = 1024**3
    logger.debug(f"Disk space: required {size / gib:.1f} GiB, available {available_space / gib:.1f} GiB")
    if size <= available_space:
        return

    cached_files = [file for repo in cache_info.repos for revision in repo.revisions for file in revision.files]

    # Remove as few least recently used files as possible
    removed_files = []
    freed_space = 0
    extra_space_needed = size - available_space
    for file in sorted(cached_files, key=lambda file: file.blob_last_accessed):
        os.remove(file.file_path)  # Remove symlink
        os.remove(file.blob_path)  # Remove contents

        removed_files.append(file)
        freed_space += file.size_on_disk
        if freed_space >= extra_space_needed:
            break
    if removed_files:
        logger.info(f"Removed {len(removed_files)} files to free {freed_space / gib:.1f} GiB of disk space")
        logger.debug(f"Removed paths: {[str(file.file_path) for file in removed_files]}")

    if freed_space < extra_space_needed:
        raise RuntimeError(
            f"Insufficient disk space to load a block. Please free {(extra_space_needed - freed_space) / gib:.1f} GiB "
            f"on the volume for {cache_dir} or increase --max_disk_space if you set it manually"
        )
