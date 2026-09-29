"""Replay-scoped Filesystem MCP service for normalized MTAR workspaces."""

from __future__ import annotations

import socket
import stat
import subprocess
import time
from pathlib import Path


def prepare_filesystem_workspace(workspace: Path | str) -> None:
    """Make the isolated bind mount writable by the non-root MCP container."""

    root = Path(workspace)
    if not root.is_dir():
        raise ValueError(f"filesystem MCP workspace is not a directory: {root}")
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_symlink():
            raise ValueError(f"filesystem MCP workspace contains a symlink: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(mode | stat.S_IWGRP | stat.S_IWOTH | stat.S_IXGRP | stat.S_IXOTH)
        elif path.is_file():
            path.chmod(mode | stat.S_IWGRP | stat.S_IWOTH)


class FilesystemMCPService:
    def __init__(
        self,
        workspace: Path | str,
        *,
        environment_id: str,
        output_dir: Path | str,
        node_image: str,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.environment_id = environment_id
        self.output_dir = Path(output_dir).resolve()
        self.node_image = node_image
        self.name = ("dart-mtar-fs-" + environment_id)[-63:]
        self.port: int | None = None

    def start(self) -> str:
        self.workspace.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--label",
            f"org.sead.environment-id={self.environment_id}",
            "-p",
            "127.0.0.1::9090",
            "-v",
            f"{self.workspace}:/workspace",
            self.node_image,
            "supergateway",
            "--port",
            "9090",
            "--outputTransport",
            "streamableHttp",
            "--stateful",
            "--stdio",
            "mcp-server-filesystem /workspace /tmp /var /etc",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                completed.stderr.strip() or "filesystem MCP container failed"
            )
        port_text = subprocess.check_output(
            ["docker", "port", self.name, "9090/tcp"], text=True
        ).strip()
        self.port = int(port_text.rsplit(":", 1)[1])
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return f"http://127.0.0.1:{self.port}/mcp"
            except OSError:
                time.sleep(1)
        self.stop()
        raise RuntimeError("filesystem MCP server did not become ready")

    def stop(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logs = subprocess.run(
            ["docker", "logs", self.name], text=True, capture_output=True, check=False
        )
        (self.output_dir / "filesystem_mcp.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(
            ["docker", "rm", "-f", "-v", self.name],
            text=True,
            capture_output=True,
            check=False,
        )
