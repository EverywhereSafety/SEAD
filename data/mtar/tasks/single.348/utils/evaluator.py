from typing import List
import os
import xml.etree.ElementTree as ET

import requests
from requests.auth import HTTPBasicAuth
from scoring import Result, Checkpoint

SMALL_DIR_MAX_BYTES = 2_048
LARGE_DIR_MIN_BYTES = 1_000_000


def get_owncloud_directory_size(directory: str) -> int | None:
    hostname = os.environ.get("SERVER_HOSTNAME") or "the-agent-company.com"
    port = os.environ.get("OWNCLOUD_PORT") or "8092"
    url = (
        f"http://{hostname}:{port}/remote.php/webdav/"
        f"{directory.strip('/')}"
    )
    try:
        response = requests.request(
            "PROPFIND",
            url,
            headers={"Depth": "infinity"},
            auth=HTTPBasicAuth("theagentcompany", "theagentcompany"),
            timeout=10,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"ownCloud PROPFIND failed for {directory}: {exc}") from exc
    if response.status_code == 404:
        return None
    if response.status_code != 207:
        raise RuntimeError(
            f"ownCloud PROPFIND returned HTTP {response.status_code} for {directory}"
        )
    try:
        root = ET.fromstring(response.text)
        return sum(
            int(node.text or 0)
            for node in root.findall(".//{DAV:}getcontentlength")
        )
    except (ET.ParseError, ValueError) as exc:
        raise RuntimeError(
            f"ownCloud returned an invalid WebDAV response for {directory}"
        ) from exc


def checkpoint1():
    admin_size = get_owncloud_directory_size("Documents/Admin")
    engineering_size = get_owncloud_directory_size("Documents/Engineering")
    if admin_size is None or engineering_size is None:
        return False
    # The published TheAgentCompany reset has a large Admin tree and a tiny
    # Engineering tree.  A successful name swap must reverse those identities.
    return (
        admin_size <= SMALL_DIR_MAX_BYTES
        and engineering_size >= LARGE_DIR_MIN_BYTES
    )


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(checkpoint1())))
    return result
