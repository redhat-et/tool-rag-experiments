from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import shutil
import tarfile
import zipfile
from typing import Tuple, List

from pydantic import AnyUrl

from evaluator.utils.utils import log_verbose

_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".tbz2",
    ".txz",
    ".tar",
    ".zip",
)


def _split_archive_basename(filename: str) -> Tuple[str, str]:
    """
    Returns (stem_without_archive_suffix, archive_suffix) where suffix is one of _ARCHIVE_SUFFIXES
    or ("filename", "") if none matched. Longest-suffix wins to handle .tar.gz correctly.
    """
    lname = filename.lower()
    for suf in _ARCHIVE_SUFFIXES:
        if lname.endswith(suf):
            return filename[: -len(suf)], filename[-len(suf):]
    return filename, ""


def _download(url: str, dest_path: Path) -> None:
    """
    Stream-download to dest_path (overwrite if exists).
    """
    print(f"Downloading the dataset from {url}...")
    with urlopen(url) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"Dataset successfully downloaded to {dest_path}")


def fetch_remote_path(remote_path: AnyUrl, local_dir: str | Path) -> Path:
    """
    Download a remote file or archive into `local_dir` if needed, and return the local path.

    If remote is a non-archive file:
      - If a same-named file already exists in `local_dir`, do nothing and return that path.
      - Otherwise download it to `local_dir/<filename>` and return that file path.

    If remote is an archive:
      - Define target directory as the archive name without its archive extension.
      - If `local_dir/<name_without_ext>` already exists (as a directory), do nothing and return it.
      - Otherwise download the archive to `local_dir/<filename>`, extract it into `local_dir`,
        delete the archive, and return `local_dir/<name_without_ext>`.

    Assumes an archive contains a top-level directory matching the archive name (minus extension).
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Derive filename from the remote path (URL or path-like string).
    # We only use the last path component.
    filename = Path(remote_path.path).name
    if not filename:
        raise ValueError("Remote path must include a filename component.")

    is_archive = any(filename.lower().endswith(suf) for suf in _ARCHIVE_SUFFIXES)

    if not is_archive:
        target_file = local_dir / filename
        if target_file.exists():
            log_verbose(f"Dataset file {target_file} seems to already exist, skipping the dataset download step")
            return target_file
        # Download
        try:
            _download(str(remote_path), target_file)
        except (HTTPError, URLError) as e:
            # Clean up partial
            if target_file.exists():
                try:
                    target_file.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Failed to download file: {e}") from e
        return target_file

    # Archive case
    base_name, _ = _split_archive_basename(filename)
    target_dir = local_dir / base_name
    if target_dir.is_dir():
        log_verbose(f"Dataset directory {target_dir} seems to already exist, skipping the dataset download step")
        return target_dir

    archive_path = local_dir / filename
    # Download archive
    try:
        _download(str(remote_path), archive_path)
    except (HTTPError, URLError) as e:
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to download archive: {e}") from e

    # Extract
    # We assume the downloaded archive to contain a single directory named identically to the archive,
    # i.e., with the name equal to base_name, thus making its local path identical to target_dir.
    print(f"Extracting dataset to {target_dir}...")
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(local_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                tf.extractall(local_dir)
        else:
            raise RuntimeError(f"Unsupported or corrupt archive format: {archive_path.name}")
    except Exception:
        # Keep archive for debugging if extraction fails
        raise
    else:
        # Delete archive after successful extraction
        try:
            archive_path.unlink()
        except OSError:
            pass

    # Return the directory assumed to be created
    print(f"Dataset extracted successfully.")
    return target_dir


def fetch_remote_paths(remote_paths: List[AnyUrl], local_dir: str | Path) -> List[Path]:
    local_paths = []
    for remote_path in remote_paths:
        local_path = fetch_remote_path(remote_path, local_dir)
        local_paths.append(local_path)
    return local_paths
