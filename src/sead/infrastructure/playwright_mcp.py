"""Replay-scoped Playwright MCP service for normalized MTAR tasks."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path


class PlaywrightMCPService:
    """Run one browser server with the replay workspace mounted at /workspace."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        environment_id: str,
        output_dir: Path | str,
        image: str,
        port: int = 9092,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.environment_id = environment_id
        self.output_dir = Path(output_dir).resolve()
        self.image = image
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Playwright MCP port must be an integer from 1 to 65535")
        self.port = port
        self.name = ("dart-playwright-" + environment_id)[-63:]

    def start(self) -> str:
        # Callers hold the service resource lock. Refuse an occupied endpoint
        # rather than mistaking a stale/foreign browser for this replay's server.
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", self.port))
        except OSError as exc:
            raise RuntimeError(f"Playwright MCP port {self.port} is unavailable") from exc
        self.workspace.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--label",
            f"org.sead.environment-id={self.environment_id}",
            "--network",
            "host",
            "--add-host",
            "the-agent-company.com:127.0.0.1",
            "-e",
            f"MCP_PLAYWRIGHT_PORT={self.port}",
            "-v",
            f"{self.workspace}:/workspace",
            self.image,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                completed.stderr.strip() or "Playwright MCP container failed"
            )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    # Playwright's Streamable HTTP endpoint expires an idle
                    # session while the Target model is thinking.  The legacy
                    # SSE endpoint keeps the browser session bound to the open
                    # connection, so page/login state survives long turns.
                    return f"http://127.0.0.1:{self.port}/sse"
            except OSError:
                time.sleep(1)
        self.stop()
        raise RuntimeError("Playwright MCP server did not become ready")

    def stop(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logs = subprocess.run(
            ["docker", "logs", self.name], text=True, capture_output=True, check=False
        )
        (self.output_dir / "playwright_mcp.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        subprocess.run(
            ["docker", "rm", "-f", "-v", self.name],
            text=True,
            capture_output=True,
            check=False,
        )
