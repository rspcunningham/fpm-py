from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from threading import Lock
from typing import BinaryIO, Callable, Protocol, cast

from tqdm.auto import tqdm

from ptych.data.parse import parse_manifest


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_DOWNLOAD_WORKERS = 6
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)
PROPFIND_DIRECTORY_SIZE_BODY = """<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop>
    <oc:size />
  </d:prop>
</d:propfind>
""".encode("utf-8")
WEBDAV_NAMESPACES = {"oc": "http://owncloud.org/ns"}
DAV_NAMESPACES = {"d": "DAV:"}


class DownloadResponse(Protocol):
    headers: Message

    def read(self, size: int = -1) -> bytes: ...


class DownloadResponseContext(DownloadResponse, Protocol):
    def __enter__(self) -> DownloadResponse: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> object: ...


class ProgressBar(Protocol):
    def update(self, n: int = 1) -> object: ...


def _auth_header(share_id: str) -> str:
    credentials = f"{share_id}:".encode("utf-8")
    return "Basic " + base64.b64encode(credentials).decode("ascii")


def _dav_url(base_url: str, share_id: str, path: str = "") -> str:
    base = base_url.rstrip("/")
    encoded_segments = [
        urllib.parse.quote(part, safe="") for part in path.split("/") if part
    ]
    suffix = "/" + "/".join(encoded_segments) if encoded_segments else ""
    return f"{base}/public.php/dav/files/{urllib.parse.quote(share_id, safe='')}{suffix}/"


def _dav_file_url(base_url: str, share_id: str, path: str) -> str:
    return _dav_url(base_url, share_id, path).rstrip("/")


def _response_content_length(response: DownloadResponse) -> int | None:
    raw_value = response.headers.get("Content-Length")
    if raw_value is None:
        return None

    try:
        content_length = int(raw_value)
    except ValueError:
        return None

    return content_length if content_length >= 0 else None


def _open_response(request: urllib.request.Request) -> DownloadResponseContext:
    return cast(DownloadResponseContext, urllib.request.urlopen(request))


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)}{unit}"
    return f"{value:.2f}{unit}"


def _download_workers() -> int:
    raw_value = os.environ.get("PTYCH_DOWNLOAD_WORKERS")
    if raw_value is None:
        return DEFAULT_DOWNLOAD_WORKERS

    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_DOWNLOAD_WORKERS

    return max(1, parsed)


def _download_file_to_path(
    url: str,
    share_id: str,
    destination: Path,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> int:
    request = urllib.request.Request(
        url,
        headers={"Authorization": _auth_header(share_id)},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    with _open_response(request) as response, destination.open("wb") as fh:
        while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
            fh.write(chunk)
            bytes_written += len(chunk)
            if progress_callback is not None:
                progress_callback(len(chunk))

    return bytes_written


def _safe_destination(root: Path, relative_path: str) -> Path:
    destination_root = root.resolve()
    destination = (destination_root / relative_path).resolve()
    if destination == destination_root or not destination.is_relative_to(destination_root):
        raise RuntimeError(f"Invalid capture path '{relative_path}'")
    return destination


def _zip_content_length(url: str, share_id: str) -> int | None:
    request = urllib.request.Request(
        url,
        headers={"Authorization": _auth_header(share_id)},
        method="HEAD",
    )
    try:
        with _open_response(request) as response:
            return _response_content_length(response)
    except urllib.error.URLError:
        return None


def _directory_size_hint(base_url: str, share_id: str, directory_name: str) -> int | None:
    request = urllib.request.Request(
        _dav_url(base_url, share_id, directory_name),
        data=PROPFIND_DIRECTORY_SIZE_BODY,
        headers={
            "Authorization": _auth_header(share_id),
            "Content-Type": "application/xml",
            "Depth": "0",
        },
        method="PROPFIND",
    )
    try:
        with _open_response(request) as response:
            payload = response.read()
    except urllib.error.URLError:
        return None

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return None

    raw_size = root.findtext(".//oc:size", namespaces=WEBDAV_NAMESPACES)
    if raw_size is None:
        return None

    try:
        size = int(raw_size)
    except ValueError:
        return None

    return size if size >= 0 else None
@dataclass(frozen=True)
class RemoteFileEntry:
    path: str
    size: int | None


def _list_directory_files(base_url: str, share_id: str, directory_path: str) -> list[RemoteFileEntry]:
    normalized_dir = directory_path.strip("/")
    request = urllib.request.Request(
        _dav_url(base_url, share_id, normalized_dir),
        data=b"""<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:getcontentlength />
  </d:prop>
</d:propfind>
""",
        headers={
            "Authorization": _auth_header(share_id),
            "Content-Type": "application/xml",
            "Depth": "1",
        },
        method="PROPFIND",
    )

    with _open_response(request) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    entries: list[RemoteFileEntry] = []
    prefix = (
        f"/public.php/dav/files/{urllib.parse.quote(share_id, safe='')}/{normalized_dir}/"
    )
    for response_el in root.findall("d:response", DAV_NAMESPACES):
        href = response_el.findtext("d:href", namespaces=DAV_NAMESPACES)
        if href is None:
            continue

        decoded_href = urllib.parse.unquote(urllib.parse.urlparse(href).path)
        if decoded_href.endswith("/"):
            continue
        if not decoded_href.startswith(prefix):
            continue

        relative_path = decoded_href.removeprefix(prefix)
        if not relative_path:
            continue

        raw_size = response_el.findtext(".//d:getcontentlength", namespaces=DAV_NAMESPACES)
        size: int | None
        if raw_size is None:
            size = None
        else:
            try:
                size = int(raw_size)
            except ValueError:
                size = None

        entries.append(RemoteFileEntry(path=relative_path, size=size))

    return entries


def _copy_stream_with_progress(response: DownloadResponse, fh: BinaryIO, progress: ProgressBar) -> None:
    while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
        fh.write(chunk)
        progress.update(len(chunk))


def _stream_download_to_file(
    response: DownloadResponse,
    destination: Path,
    description: str,
    *,
    total_bytes: int | None = None,
) -> None:
    total_bytes = total_bytes if total_bytes is not None else _response_content_length(response)
    with destination.open("wb") as fh:
        if total_bytes is None:
            with tqdm(
                desc=description,
                dynamic_ncols=True,
                unit="B",
                unit_divisor=1024,
                unit_scale=True,
            ) as progress:
                _copy_stream_with_progress(response, fh, progress)
            return

        with tqdm(
            total=total_bytes,
            desc=description,
            dynamic_ncols=True,
            unit="B",
            unit_divisor=1024,
            unit_scale=True,
            bar_format=PROGRESS_BAR_FORMAT,
        ) as progress:
            _copy_stream_with_progress(response, fh, progress)


def download_directory_zip_to_path(
    base_url: str,
    share_id: str,
    directory_name: str,
    destination_path: str | Path,
) -> Path:
    """Download a directory from a public Nextcloud share as a ZIP file."""
    normalized_dir = directory_name.strip("/")
    if not normalized_dir:
        raise ValueError("directory_name must not be empty")

    output_path = Path(destination_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zip_url = _dav_url(base_url, share_id, normalized_dir) + "?accept=zip"
    zip_total_bytes = _zip_content_length(zip_url, share_id)
    dataset_size_hint = None
    if zip_total_bytes is None:
        dataset_size_hint = _directory_size_hint(base_url, share_id, normalized_dir)

    description = f"Downloading {Path(normalized_dir).name}"
    if dataset_size_hint is not None:
        description = f"{description} ({_format_bytes(dataset_size_hint)} dataset)"

    request = urllib.request.Request(
        zip_url,
        headers={"Authorization": _auth_header(share_id)},
    )

    try:
        with _open_response(request) as response:
            _stream_download_to_file(
                response,
                output_path,
                description=description,
                total_bytes=zip_total_bytes,
            )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ZIP download failed with HTTP {exc.code}") from exc

    return output_path


@dataclass(frozen=True)
class NextcloudShareTransport:
    base_url: str
    share_id: str

    def download_dataset(self, dataset_id: str, destination_path: str | Path) -> Path:
        dataset_root = Path(destination_path)
        dataset_root.mkdir(parents=True, exist_ok=True)

        manifest_remote_path = f"{dataset_id}/info.json"
        manifest_local_path = dataset_root / "info.json"
        _download_file_to_path(
            _dav_file_url(self.base_url, self.share_id, manifest_remote_path),
            self.share_id,
            manifest_local_path,
        )

        with manifest_local_path.open(encoding="utf-8") as fh:
            manifest_data = cast(dict[str, object], json.load(fh))
        manifest = parse_manifest(manifest_data)

        capture_entries = {
            entry.path: entry
            for entry in _list_directory_files(
                self.base_url,
                self.share_id,
                f"{dataset_id}/captures",
            )
        }

        capture_files = [capture.filename for capture in manifest.captures]
        missing_files = [filename for filename in capture_files if filename not in capture_entries]
        if missing_files:
            first_missing = missing_files[0]
            raise RuntimeError(
                f"Dataset '{dataset_id}' is missing capture file '{first_missing}' on the share"
            )

        if len(set(capture_files)) != len(capture_files):
            raise RuntimeError(f"Dataset '{dataset_id}' contains duplicate capture filenames")

        total_bytes: int | None
        if all(capture_entries[filename].size is not None for filename in capture_files):
            total_bytes = sum(
                cast(int, capture_entries[filename].size)
                for filename in capture_files
            )
        else:
            total_bytes = None

        description = f"Downloading {dataset_id}"
        if total_bytes is not None:
            description = (
                f"{description} ({_format_bytes(total_bytes)} across {len(capture_files)} files)"
            )

        captures_dir = dataset_root / "captures"
        progress_lock = Lock()

        def update_progress(delta: int) -> None:
            with progress_lock:
                progress.update(delta)

        max_workers = min(_download_workers(), len(capture_files)) or 1
        with tqdm(
            total=total_bytes,
            desc=description,
            dynamic_ncols=True,
            unit="B",
            unit_divisor=1024,
            unit_scale=True,
            bar_format=PROGRESS_BAR_FORMAT if total_bytes is not None else None,
        ) as progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _download_file_to_path,
                        _dav_file_url(
                            self.base_url,
                            self.share_id,
                            f"{dataset_id}/captures/{filename}",
                        ),
                        self.share_id,
                        _safe_destination(captures_dir, filename),
                        progress_callback=update_progress,
                    )
                    for filename in capture_files
                ]
                for future in futures:
                    future.result()

        return dataset_root

    def download_directory_zip(self, dataset_id: str, destination_path: str | Path) -> Path:
        return download_directory_zip_to_path(
            self.base_url,
            self.share_id,
            dataset_id,
            destination_path,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a directory from a public Nextcloud share")
    parser.add_argument("base_url")
    parser.add_argument("share_id")
    parser.add_argument("directory_name")
    parser.add_argument(
        "--destination-path",
        default=None,
        help="Explicit path where the .zip file will be written",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    base_url = cast(str, args.base_url)
    share_id = cast(str, args.share_id)
    directory_name = cast(str, args.directory_name)
    destination_path = cast(str | None, args.destination_path)

    if destination_path is None:
        destination_path = f"{Path(directory_name).name}.zip"

    output_path = download_directory_zip_to_path(
        base_url,
        share_id,
        directory_name,
        destination_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
