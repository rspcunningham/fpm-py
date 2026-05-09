from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import cast

from ptych.data.upload.nextcloud_webdav import (
    AUTH_CONFIG_PATH,
    NEXTCLOUD_REMOTE_ROOT,
    NextcloudAuthError,
    NextcloudCredentials,
    NextcloudWebDAVClient,
    NextcloudWebDAVError,
    format_bytes,
    list_public_datasets,
    read_credentials,
    verify_public_manifest,
    write_credentials,
)
from ptych.data.upload.validate import (
    DatasetValidationError,
    normalize_dataset_id,
    validate_dataset,
)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except (DatasetValidationError, NextcloudWebDAVError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt as exc:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptych-dataset",
        description="Validate and publish ptych datasets to the project Nextcloud bucket",
    )
    subparsers = parser.add_subparsers(required=True)

    auth_parser = subparsers.add_parser(
        "auth",
        help="Store Nextcloud app-password credentials",
    )
    auth_parser.set_defaults(func=_auth)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a local dataset directory",
    )
    validate_parser.add_argument("dataset_dir", type=Path)
    validate_parser.set_defaults(func=_validate)

    list_parser = subparsers.add_parser(
        "list",
        help="List datasets currently visible on the public share",
    )
    list_parser.set_defaults(func=_list)

    push_parser = subparsers.add_parser(
        "push",
        help="Validate and upload a local dataset directory",
    )
    push_parser.add_argument("dataset_dir", type=Path)
    push_parser.add_argument(
        "--name",
        required=True,
        help="Remote dataset name used by PtychStudy.load",
    )
    push_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing remote dataset with the same name",
    )
    push_parser.set_defaults(func=_push)

    return parser


def _auth(_args: argparse.Namespace) -> None:
    username = input("Nextcloud username: ").strip()
    app_password = getpass.getpass("Nextcloud app password: ")
    if not username:
        raise NextcloudAuthError("Nextcloud username must not be empty")
    if not app_password:
        raise NextcloudAuthError("Nextcloud app password must not be empty")

    credentials = NextcloudCredentials(
        username=username,
        app_password=app_password,
    )
    client = NextcloudWebDAVClient(credentials)
    root_entry = client.check_remote_root()
    path = write_credentials(credentials)

    permissions = (
        f", permissions {root_entry.permissions}" if root_entry.permissions else ""
    )
    print(
        f"Authenticated for {NEXTCLOUD_REMOTE_ROOT}{permissions}. "
        f"Credentials saved to {path}."
    )


def _validate(args: argparse.Namespace) -> None:
    dataset_dir = cast(Path, args.dataset_dir)
    dataset = validate_dataset(dataset_dir)
    print(
        f"Valid dataset: {dataset.root} "
        f"({len(dataset.capture_paths)} captures, {format_bytes(dataset.total_bytes)})"
    )


def _list(_args: argparse.Namespace) -> None:
    entries = list_public_datasets()
    if not entries:
        print("No public datasets found.")
        return

    for entry in entries:
        size = format_bytes(entry.size) if entry.size is not None else "unknown size"
        print(f"{entry.name}\t{size}")


def _push(args: argparse.Namespace) -> None:
    dataset_dir = cast(Path, args.dataset_dir)
    dataset_id = normalize_dataset_id(cast(str, args.name))
    overwrite = cast(bool, args.overwrite)
    dataset = validate_dataset(dataset_dir)

    try:
        credentials = read_credentials()
    except NextcloudAuthError as exc:
        raise NextcloudAuthError(f"{exc} Credentials path: {AUTH_CONFIG_PATH}") from exc

    client = NextcloudWebDAVClient(credentials)
    client.check_remote_root()
    client.upload_dataset(dataset, dataset_id, overwrite=overwrite)
    verify_public_manifest(dataset_id)
    print(f"Published dataset '{dataset_id}'.")


if __name__ == "__main__":
    main()
