from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from tqdm.auto import tqdm

from ptych.data.upload.validate import ValidatedDataset


NEXTCLOUD_BASE_URL = "https://dqe.asuscomm.com"
NEXTCLOUD_PUBLIC_SHARE_ID = "SLbNBTqK9firqZM"
NEXTCLOUD_REMOTE_ROOT = "public_fpm_data"
AUTH_CONFIG_PATH = Path.home() / ".config" / "ptych" / "nextcloud-auth.json"
UPLOAD_CHUNK_SIZE = 1024 * 1024
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)
DAV_NAMESPACES = {"d": "DAV:", "oc": "http://owncloud.org/ns"}
PROPFIND_BODY = b"""<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop>
    <d:displayname />
    <d:resourcetype />
    <d:getcontentlength />
    <oc:size />
    <oc:permissions />
  </d:prop>
</d:propfind>
"""


class ProgressBar(Protocol):
    def update(self, n: int = 1) -> object: ...


class NextcloudWebDAVError(Exception):
    pass


class NextcloudAuthError(NextcloudWebDAVError):
    pass


class NextcloudHTTPError(NextcloudWebDAVError):
    code: int

    def __init__(self, method: str, url: str, code: int, detail: str) -> None:
        self.code = code
        super().__init__(f"{method} {url} failed with HTTP {code}: {detail}")


@dataclass(frozen=True)
class NextcloudCredentials:
    username: str
    app_password: str


@dataclass(frozen=True)
class WebDAVEntry:
    relative_path: str
    name: str
    is_collection: bool
    size: int | None
    permissions: str | None


class _ProgressReader:
    def __init__(self, fh: BinaryIO, progress: ProgressBar) -> None:
        self.fh = fh
        self.progress = progress

    def read(self, size: int = -1) -> bytes:
        chunk = self.fh.read(size)
        self.progress.update(len(chunk))
        return chunk


def read_credentials(path: Path = AUTH_CONFIG_PATH) -> NextcloudCredentials:
    if not path.is_file():
        raise NextcloudAuthError(
            "Nextcloud credentials are not configured. Run `ptych-dataset auth` first."
        )

    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise NextcloudAuthError(f"Invalid credentials file: {path}") from exc

    if not isinstance(data, dict):
        raise NextcloudAuthError(f"Invalid credentials file: {path}")

    username = data.get("username")
    app_password = data.get("app_password")
    if not isinstance(username, str) or not username:
        raise NextcloudAuthError(f"Credentials file is missing username: {path}")
    if not isinstance(app_password, str) or not app_password:
        raise NextcloudAuthError(f"Credentials file is missing app_password: {path}")

    return NextcloudCredentials(username=username, app_password=app_password)


def write_credentials(
    credentials: NextcloudCredentials, path: Path = AUTH_CONFIG_PATH
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "username": credentials.username,
                    "app_password": credentials.app_password,
                },
                fh,
                indent=2,
            )
            fh.write("\n")
    finally:
        os.chmod(path, 0o600)
    return path


def list_public_datasets() -> list[WebDAVEntry]:
    request = urllib.request.Request(
        _public_url("", collection=True),
        data=PROPFIND_BODY,
        headers={
            "Authorization": _basic_auth_header(NEXTCLOUD_PUBLIC_SHARE_ID, ""),
            "Content-Type": "application/xml",
            "Depth": "1",
        },
        method="PROPFIND",
    )
    payload = _open_public_response(request)
    entries = _parse_propfind_response(
        payload,
        base_prefix=f"/public.php/dav/files/{urllib.parse.quote(NEXTCLOUD_PUBLIC_SHARE_ID, safe='')}/",
    )
    return sorted(
        [
            entry
            for entry in entries
            if entry.relative_path and entry.is_collection and entry.name != ".staging"
        ],
        key=lambda entry: entry.name,
    )


def verify_public_manifest(dataset_id: str) -> None:
    request = urllib.request.Request(
        _public_url(f"{dataset_id}/info.json"),
        headers={"Authorization": _basic_auth_header(NEXTCLOUD_PUBLIC_SHARE_ID, "")},
    )
    _open_public_response(request)


def format_bytes(num_bytes: int) -> str:
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


@dataclass
class NextcloudWebDAVClient:
    credentials: NextcloudCredentials
    base_url: str = NEXTCLOUD_BASE_URL

    def check_remote_root(self) -> WebDAVEntry:
        entries = self.propfind(NEXTCLOUD_REMOTE_ROOT, collection=True)
        if not entries:
            raise NextcloudWebDAVError(
                f"Nextcloud folder '{NEXTCLOUD_REMOTE_ROOT}' was not found"
            )
        entry = entries[0]
        if not entry.is_collection:
            raise NextcloudWebDAVError(
                f"Nextcloud path '{NEXTCLOUD_REMOTE_ROOT}' is not a folder"
            )
        return entry

    def upload_dataset(
        self,
        dataset: ValidatedDataset,
        dataset_id: str,
        *,
        overwrite: bool = False,
    ) -> None:
        final_path = _remote_join(NEXTCLOUD_REMOTE_ROOT, dataset_id)
        staging_parent = _remote_join(NEXTCLOUD_REMOTE_ROOT, ".staging")
        staging_path = _remote_join(staging_parent, f"{dataset_id}-{uuid.uuid4().hex}")
        staging_captures_path = _remote_join(staging_path, "captures")

        if self.exists(final_path):
            if not overwrite:
                raise NextcloudWebDAVError(
                    f"Remote dataset '{dataset_id}' already exists. Use --overwrite to replace it."
                )

        try:
            self.mkcol(staging_parent)
            self.mkcol(staging_path)
            self.mkcol(staging_captures_path)

            with tqdm(
                total=dataset.total_bytes,
                desc=f"Uploading {dataset_id}",
                dynamic_ncols=True,
                unit="B",
                unit_divisor=1024,
                unit_scale=True,
                bar_format=PROGRESS_BAR_FORMAT,
            ) as progress:
                for capture_path in dataset.capture_paths:
                    remote_path = _remote_join(
                        staging_captures_path,
                        capture_path.name,
                    )
                    self.put_file(capture_path, remote_path, progress)

                self.put_file(
                    dataset.root / "info.json",
                    _remote_join(staging_path, "info.json"),
                    progress,
                )

            for capture_path in dataset.capture_paths:
                self.require_file_size(
                    _remote_join(staging_captures_path, capture_path.name),
                    capture_path.stat().st_size,
                )
            self.require_file_size(
                _remote_join(staging_path, "info.json"),
                (dataset.root / "info.json").stat().st_size,
            )

            if overwrite:
                self.delete(final_path)
            self.move(staging_path, final_path)
        except Exception:
            self.delete(staging_path)
            raise

    def propfind(
        self,
        remote_path: str,
        *,
        depth: str = "0",
        collection: bool = False,
    ) -> list[WebDAVEntry]:
        payload = self._request(
            "PROPFIND",
            remote_path,
            data=PROPFIND_BODY,
            headers={
                "Content-Type": "application/xml",
                "Depth": depth,
            },
            collection=collection,
        )
        return _parse_propfind_response(
            payload,
            base_prefix=f"/remote.php/dav/files/{urllib.parse.quote(self.credentials.username, safe='')}/",
        )

    def exists(self, remote_path: str) -> bool:
        try:
            self.propfind(remote_path)
        except NextcloudHTTPError as exc:
            if exc.code == 404:
                return False
            raise
        return True

    def mkcol(self, remote_path: str) -> None:
        try:
            self._request("MKCOL", remote_path, collection=True)
        except NextcloudHTTPError as exc:
            if exc.code == 405:
                return
            raise

    def delete(self, remote_path: str) -> None:
        try:
            self._request("DELETE", remote_path)
        except NextcloudHTTPError as exc:
            if exc.code == 404:
                return
            raise

    def move(self, source_path: str, destination_path: str) -> None:
        self._request(
            "MOVE",
            source_path,
            headers={
                "Destination": self._url(destination_path),
                "Overwrite": "F",
            },
        )

    def put_file(
        self,
        local_path: Path,
        remote_path: str,
        progress: ProgressBar,
    ) -> None:
        size = local_path.stat().st_size
        with local_path.open("rb") as fh:
            reader = _ProgressReader(fh, progress)
            self._request(
                "PUT",
                remote_path,
                data=reader,
                headers={
                    "Content-Length": str(size),
                    "Content-Type": "application/octet-stream",
                },
            )

    def require_file_size(self, remote_path: str, expected_size: int) -> None:
        entries = self.propfind(remote_path)
        if not entries or entries[0].size is None:
            raise NextcloudWebDAVError(
                f"Could not verify uploaded file size for {remote_path}"
            )
        actual_size = entries[0].size
        if actual_size != expected_size:
            raise NextcloudWebDAVError(
                f"Uploaded file size mismatch for {remote_path}: expected {expected_size}, got {actual_size}"
            )

    def _request(
        self,
        method: str,
        remote_path: str,
        *,
        data: object | None = None,
        headers: dict[str, str] | None = None,
        collection: bool = False,
    ) -> bytes:
        url = self._url(remote_path, collection=collection)
        request_headers = {
            "Authorization": _basic_auth_header(
                self.credentials.username,
                self.credentials.app_password,
            )
        }
        if headers:
            request_headers.update(headers)

        request = urllib.request.Request(
            url,
            data=cast(bytes | None, data),
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace").strip()
            if not detail:
                detail = exc.reason
            raise NextcloudHTTPError(method, url, exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise NextcloudWebDAVError(f"{method} {url} failed: {exc}") from exc

    def _url(self, remote_path: str, *, collection: bool = False) -> str:
        return _private_url(
            self.base_url,
            self.credentials.username,
            remote_path,
            collection=collection,
        )


def _open_public_response(request: urllib.request.Request) -> bytes:
    try:
        with urllib.request.urlopen(request) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace").strip()
        if not detail:
            detail = exc.reason
        raise NextcloudHTTPError(
            request.get_method(), request.full_url, exc.code, detail
        ) from exc
    except urllib.error.URLError as exc:
        raise NextcloudWebDAVError(
            f"{request.get_method()} {request.full_url} failed: {exc}"
        ) from exc


def _parse_propfind_response(payload: bytes, base_prefix: str) -> list[WebDAVEntry]:
    root = ET.fromstring(payload)
    entries: list[WebDAVEntry] = []
    for response_el in root.findall("d:response", DAV_NAMESPACES):
        href = response_el.findtext("d:href", namespaces=DAV_NAMESPACES)
        prop = _successful_prop(response_el)
        if href is None or prop is None:
            continue

        decoded_path = urllib.parse.unquote(urllib.parse.urlparse(href).path)
        relative_path = _relative_dav_path(decoded_path, base_prefix)
        display_name = prop.findtext("d:displayname", namespaces=DAV_NAMESPACES)
        name = display_name or Path(relative_path.rstrip("/")).name
        resource_type = prop.find("d:resourcetype", DAV_NAMESPACES)
        is_collection = (
            resource_type is not None
            and resource_type.find("d:collection", DAV_NAMESPACES) is not None
        )
        size = _parse_optional_int(
            prop.findtext("d:getcontentlength", namespaces=DAV_NAMESPACES)
            or prop.findtext("oc:size", namespaces=DAV_NAMESPACES)
        )
        permissions = prop.findtext("oc:permissions", namespaces=DAV_NAMESPACES)
        entries.append(
            WebDAVEntry(
                relative_path=relative_path,
                name=name,
                is_collection=is_collection,
                size=size,
                permissions=permissions,
            )
        )
    return entries


def _successful_prop(response_el: ET.Element) -> ET.Element | None:
    for propstat in response_el.findall("d:propstat", DAV_NAMESPACES):
        status = propstat.findtext("d:status", namespaces=DAV_NAMESPACES)
        if status is None or " 200 " not in status:
            continue
        return propstat.find("d:prop", DAV_NAMESPACES)
    return None


def _relative_dav_path(decoded_path: str, base_prefix: str) -> str:
    prefix = urllib.parse.unquote(base_prefix)
    if decoded_path == prefix.rstrip("/") or decoded_path == prefix:
        return ""
    if decoded_path.startswith(prefix):
        return decoded_path.removeprefix(prefix).strip("/")
    return Path(decoded_path.rstrip("/")).name


def _parse_optional_int(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def _remote_join(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _private_url(
    base_url: str,
    username: str,
    remote_path: str,
    *,
    collection: bool = False,
) -> str:
    path = _encoded_path(remote_path)
    suffix = f"/{path}" if path else ""
    url = (
        f"{base_url.rstrip('/')}/remote.php/dav/files/"
        f"{urllib.parse.quote(username, safe='')}{suffix}"
    )
    if collection and not url.endswith("/"):
        url += "/"
    return url


def _public_url(remote_path: str, *, collection: bool = False) -> str:
    path = _encoded_path(remote_path)
    suffix = f"/{path}" if path else ""
    url = (
        f"{NEXTCLOUD_BASE_URL.rstrip('/')}/public.php/dav/files/"
        f"{urllib.parse.quote(NEXTCLOUD_PUBLIC_SHARE_ID, safe='')}{suffix}"
    )
    if collection and not url.endswith("/"):
        url += "/"
    return url


def _encoded_path(remote_path: str) -> str:
    return "/".join(
        urllib.parse.quote(part, safe="") for part in remote_path.split("/") if part
    )


def _basic_auth_header(username: str, password: str) -> str:
    credentials = f"{username}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(credentials).decode("ascii")
