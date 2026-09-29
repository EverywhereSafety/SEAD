# Adapted from OpenAgentSafety / TheAgentCompany evaluation utilities.
# Copyright (c) 2024 TheAgentCompany
# SPDX-License-Identifier: MIT
# See licenses/OpenAgentSafety-MIT.txt and licenses/README.md.
# SEAD modifications: deterministic compatibility subset and runtime decoupling.

"""Deterministic import-compatible subset of OAS ``common.py``.

It implements only the filesystem-independent service probes used by the
frozen SEAD selections. LLM-backed grading remains deliberately disabled.
"""

from __future__ import annotations

import functools
import logging
import os
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Callable

import requests
from requests.auth import HTTPBasicAuth

SERVER_HOSTNAME = os.environ.get("SERVER_HOSTNAME", "the-agent-company.com")
GITLAB_BASEURL = f"http://{SERVER_HOSTNAME}:{os.environ.get('GITLAB_PORT', '8929')}"
GITLAB_HEADERS = {"PRIVATE-TOKEN": os.environ.get("GITLAB_ACCESS_TOKEN", "root-token")}
GITLAB_USER = "root"
OWNCLOUD_URL = f"http://{SERVER_HOSTNAME}:{os.environ.get('OWNCLOUD_PORT', '8092')}"
OWNCLOUD_USERNAME = "theagentcompany"
OWNCLOUD_PASSWORD = "theagentcompany"
PLANE_BASEURL = f"http://{SERVER_HOSTNAME}:{os.environ.get('PLANE_PORT', '8091')}"
PLANE_WORKSPACE_SLUG = os.environ.get("PLANE_WORKSPACE_SLUG", "tac")
PLANE_HEADERS = {
    "x-api-key": os.environ.get(
        "PLANE_API_KEY", "plane_api_83f868352c6f490aba59b869ffdae1cf"
    ),
    "Content-Type": "application/json",
}


def grader(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = func(*args, **kwargs)
            if result is None:
                logging.warning("Grader returns None, using False instead")
                return False
            return result
        except Exception as exc:  # noqa: BLE001 - matches the released boundary
            logging.error("Error in %s: %s", func.__name__, exc)
            return False

    return wrapper


def evaluate_with_llm(*args: Any, **kwargs: Any) -> bool:
    """Compatibility hook; selected evaluators import but never call it."""

    raise RuntimeError("LLM evaluation is unavailable for deterministic OAS subsets")


def make_gitlab_request(
    project_identifier: str | None = None,
    additional_path: str | None = None,
    method: str = "GET",
    params: dict[str, Any] | None = None,
):
    url = f"{GITLAB_BASEURL}/api/v4"
    if project_identifier:
        identifier = urllib.parse.quote(project_identifier, safe="")
        url += f"/projects/{identifier}"
    if additional_path:
        url += f"/{additional_path}"
    try:
        return requests.request(
            method, url, headers=GITLAB_HEADERS, params=params, timeout=30
        )
    except requests.RequestException as exc:
        logging.error("GitLab API request failed: %s", exc)
        return None


def check_file_in_owncloud_directory(file_name: str, dir_name: str) -> bool:
    server_url = f"{OWNCLOUD_URL}/remote.php/webdav/{dir_name}"
    try:
        response = requests.request(
            "PROPFIND",
            server_url,
            headers={"Depth": "1"},
            auth=HTTPBasicAuth(OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD),
            timeout=30,
        )
    except requests.RequestException as exc:
        logging.warning("OwnCloud directory probe failed: %s", exc)
        return False
    if response.status_code != 207:
        return False
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return False
    return any(
        file_name in str(element.findtext("{DAV:}href") or "")
        for element in root.findall(".//{DAV:}response")
    )


def get_binary_file_content_owncloud(file_name: str, dir_name: str) -> bytes | None:
    url = f"{OWNCLOUD_URL}/remote.php/webdav/{dir_name}/{file_name}"
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(OWNCLOUD_USERNAME, OWNCLOUD_PASSWORD),
            timeout=30,
        )
    except requests.RequestException as exc:
        logging.warning("OwnCloud download failed: %s", exc)
        return None
    return response.content if response.status_code == 200 else None


def get_all_plane_projects() -> list[dict[str, Any]]:
    url = f"{PLANE_BASEURL}/api/v1/workspaces/{PLANE_WORKSPACE_SLUG}/projects/"
    try:
        response = requests.get(url, headers=PLANE_HEADERS, timeout=30)
        response.raise_for_status()
        return list(response.json().get("results", []))
    except (requests.RequestException, ValueError) as exc:
        logging.error("Plane project request failed: %s", exc)
        return []


def get_plane_project_id(project_name: str) -> str | None:
    for project in get_all_plane_projects():
        if project.get("name") == project_name:
            return str(project.get("id"))
    return None
