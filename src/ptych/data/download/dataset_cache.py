from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast
from uuid import uuid4

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from ptych.data.parse import ManifestParseError, parse_manifest

from .nextcloud_share import NextcloudShareTransport


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ptych" / "datasets"
DEFAULT_NEXTCLOUD_BASE_URL = "http://dqe.asuscomm.com:8080"
DEFAULT_NEXTCLOUD_SHARE_ID = "SLbNBTqK9firqZM"


class NextcloudDatasetCacheError(Exception):
    pass


class InvalidDatasetIdError(NextcloudDatasetCacheError):
    pass


class DatasetValidationError(NextcloudDatasetCacheError):
    pass


class DatasetTransport(Protocol):
    def download_dataset(self, dataset_id: str, destination_path: str | Path) -> Path: ...


class NextcloudDatasetCache:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        base_url: str | None = None,
        share_id: str | None = None,
        transport: DatasetTransport | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks_dir = self.cache_dir / ".locks"
        self._locks_dir.mkdir(parents=True, exist_ok=True)

        if transport is not None:
            self.transport = transport
            self.base_url = (base_url or "").rstrip("/")
            self.share_id = share_id or ""
        else:
            self.base_url = self._resolve_base_url(base_url)
            self.share_id = self._resolve_share_id(share_id)
            self.transport = NextcloudShareTransport(self.base_url, self.share_id)

    def fetch_dataset(self, dataset_id: str) -> Path:
        normalized_id = self._normalize_dataset_id(dataset_id)
        data_path = self._data_path(normalized_id)

        with self._dataset_lock(normalized_id):
            metadata = self._read_metadata(normalized_id)
            if metadata and metadata.get("status") == "ready" and data_path.is_dir():
                print(f"Using cached dataset {normalized_id} at {data_path}")
                return data_path

            return self._refresh_dataset(normalized_id)

    def list_cached(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for dataset_root in sorted(self.cache_dir.iterdir()):
            if not dataset_root.is_dir() or dataset_root.name == ".locks":
                continue

            metadata_path = dataset_root / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with metadata_path.open(encoding="utf-8") as fh:
                    metadata = json.load(fh)
            except json.JSONDecodeError:
                continue

            metadata["data_path"] = str(dataset_root / "data")
            results.append(metadata)

        return results

    def _refresh_dataset(self, dataset_id: str) -> Path:
        dataset_root = self._dataset_root(dataset_id)
        dataset_root.mkdir(parents=True, exist_ok=True)
        data_path = dataset_root / "data"
        backup_path = dataset_root / "data.previous"
        staging_dir = dataset_root / f".staging-{uuid4().hex}"
        staging_data_path = staging_dir / "data"

        if backup_path.exists():
            shutil.rmtree(backup_path)

        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.transport.download_dataset(dataset_id, staging_data_path)
            metadata = self._validate_extracted_dataset(staging_data_path, dataset_id)

            if data_path.exists():
                data_path.rename(backup_path)

            staging_data_path.rename(data_path)
            self._write_metadata(dataset_id, metadata)

            if backup_path.exists():
                shutil.rmtree(backup_path)

            return data_path
        except Exception as exc:
            if not data_path.exists() and backup_path.exists():
                backup_path.rename(data_path)

            if not data_path.exists():
                failed_metadata = {
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "fetched_at": datetime.now(UTC).isoformat(),
                }
                self._write_metadata(dataset_id, failed_metadata)

            raise NextcloudDatasetCacheError(
                f"Failed to fetch dataset '{dataset_id}': {exc}"
            ) from exc
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _validate_extracted_dataset(self, dataset_root: Path, dataset_id: str) -> dict[str, Any]:
        manifest_path = dataset_root / "info.json"
        if not manifest_path.is_file():
            raise DatasetValidationError(f"Dataset '{dataset_id}' is missing info.json")

        captures_dir = dataset_root / "captures"
        if not captures_dir.is_dir():
            raise DatasetValidationError(f"Dataset '{dataset_id}' is missing captures/")

        try:
            with manifest_path.open(encoding="utf-8") as fh:
                manifest_data = cast(dict[str, object], json.load(fh))
            manifest = parse_manifest(manifest_data)
        except (json.JSONDecodeError, ManifestParseError, ValueError) as exc:
            raise DatasetValidationError(
                f"Dataset '{dataset_id}' has an invalid info.json: {exc}"
            ) from exc

        if not manifest.captures:
            raise DatasetValidationError(f"Dataset '{dataset_id}' has no captures")

        filenames = [capture.filename for capture in manifest.captures]
        if len(set(filenames)) != len(filenames):
            raise DatasetValidationError(
                f"Dataset '{dataset_id}' contains duplicate capture filenames"
            )

        missing_files = [
            capture.filename
            for capture in manifest.captures
            if not (captures_dir / capture.filename).is_file()
        ]
        if missing_files:
            first_missing = missing_files[0]
            raise DatasetValidationError(
                f"Dataset '{dataset_id}' is missing capture file '{first_missing}'"
            )

        return {
            "dataset_id": dataset_id,
            "study_id": str(manifest.study_id),
            "capture_count": len(manifest.captures),
            "fetched_at": datetime.now(UTC).isoformat(),
            "status": "ready",
        }

    def _normalize_dataset_id(self, dataset_id: str) -> str:
        normalized = dataset_id.strip()
        if not normalized:
            raise InvalidDatasetIdError("dataset_id must not be empty")
        if "/" in normalized or "\\" in normalized:
            raise InvalidDatasetIdError("dataset_id must be a flat directory name")
        if normalized in {".", ".."}:
            raise InvalidDatasetIdError("dataset_id must be a normal directory name")
        return normalized

    def _dataset_root(self, dataset_id: str) -> Path:
        return self.cache_dir / dataset_id

    def _data_path(self, dataset_id: str) -> Path:
        return self._dataset_root(dataset_id) / "data"

    def _metadata_path(self, dataset_id: str) -> Path:
        return self._dataset_root(dataset_id) / "metadata.json"

    def _read_metadata(self, dataset_id: str) -> dict[str, Any] | None:
        metadata_path = self._metadata_path(dataset_id)
        if not metadata_path.exists():
            return None

        try:
            with metadata_path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return None

    def _write_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> None:
        metadata_path = self._metadata_path(dataset_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=metadata_path.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(metadata, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, metadata_path)

    def _resolve_base_url(self, base_url: str | None) -> str:
        resolved = (base_url or os.environ.get("PTYCH_NEXTCLOUD_BASE_URL") or DEFAULT_NEXTCLOUD_BASE_URL).strip()
        if not resolved:
            raise NextcloudDatasetCacheError(
                "Nextcloud base URL is not configured. Set PTYCH_NEXTCLOUD_BASE_URL or "
                "edit DEFAULT_NEXTCLOUD_BASE_URL in dataset_cache.py."
            )
        return resolved.rstrip("/")

    def _resolve_share_id(self, share_id: str | None) -> str:
        resolved = (share_id or os.environ.get("PTYCH_NEXTCLOUD_SHARE_ID") or DEFAULT_NEXTCLOUD_SHARE_ID).strip()
        if not resolved:
            raise NextcloudDatasetCacheError(
                "Nextcloud share ID is not configured. Set PTYCH_NEXTCLOUD_SHARE_ID or "
                "edit DEFAULT_NEXTCLOUD_SHARE_ID in dataset_cache.py."
            )
        return resolved

    @contextmanager
    def _dataset_lock(self, dataset_id: str):
        lock_name = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()
        lock_path = self._locks_dir / f"{lock_name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            self._lock_file(lock_file)
            try:
                yield
            finally:
                self._unlock_file(lock_file)

    def _lock_file(self, lock_file: BinaryIO) -> None:
        if os.name == "nt":
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            return

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    def _unlock_file(self, lock_file: BinaryIO) -> None:
        if os.name == "nt":
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            return

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and cache datasets from a public Nextcloud share")
    parser.add_argument("dataset_id")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--share-id", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cache = NextcloudDatasetCache(
        cache_dir=args.cache_dir,
        base_url=args.base_url,
        share_id=args.share_id,
    )
    print(cache.fetch_dataset(args.dataset_id))


if __name__ == "__main__":
    main()
