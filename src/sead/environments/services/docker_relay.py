"""Host-initiated TCP through Docker exec, without a routable host gateway.

The destination is fixed by the trusted caller, never by a network request.
This keeps OpenHands/MCP control traffic working on an isolated Docker bridge.
"""

from __future__ import annotations

import errno
import random
import select
import socketserver
import subprocess
import threading


# OpenHands reserves 30000-59999 for its execution, VS Code, and app ports.
# Binding port 0 here would use Linux's overlapping ephemeral range and race
# with OpenHands between its availability check and container publication.
_DYNAMIC_RELAY_PORT_START = 20_000
_DYNAMIC_RELAY_PORT_END = 28_999


_FORWARD = """
import os, select, socket, sys
s = socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=10)
s.settimeout(None)
while True:
    ready, _, _ = select.select([s, sys.stdin.buffer], [], [])
    if s in ready:
        data = s.recv(65536)
        if not data: break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    if sys.stdin.buffer in ready:
        data = os.read(0, 65536)
        if not data: break
        s.sendall(data)
"""

_NODE_FORWARD = """
const net = require('net');
const socket = net.connect(Number(process.argv[1]), process.argv[2]);
socket.on('connect', () => { process.stdin.pipe(socket); socket.pipe(process.stdout); });
socket.on('error', () => process.exit(1));
socket.on('close', () => process.exit(0));
"""


class DockerTCPRelay:
    def __init__(
        self,
        container: str,
        remote_port: int,
        *,
        port: int = 0,
        python: str = "python3",
        host: str = "127.0.0.1",
    ) -> None:
        self.container = container
        self.remote_port = remote_port
        self.python = python
        self.host = host
        self.stopping = threading.Event()
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        relay = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                command = [
                        "docker",
                        "exec",
                        "-i",
                        relay.container,
                    ]
                command += (["node", "-e", _NODE_FORWARD, str(relay.remote_port), relay.host]
                            if relay.python == "node" else
                            [relay.python, "-u", "-c", _FORWARD, str(relay.remote_port)])
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                with relay._lock:
                    relay._processes.add(process)
                try:
                    self.request.settimeout(30)
                    while not relay.stopping.is_set():
                        ready, _, _ = select.select(
                            [self.request, process.stdout], [], [], 0.5
                        )
                        if self.request in ready:
                            data = self.request.recv(65536)
                            if not data:
                                break
                            process.stdin.write(data)
                            process.stdin.flush()
                        if process.stdout in ready:
                            data = process.stdout.read1(65536)
                            if not data:
                                break
                            self.request.sendall(data)
                except (OSError, ValueError):
                    pass  # Client disconnect or lease revocation.
                finally:
                    process.kill()
                    process.communicate(timeout=10)
                    with relay._lock:
                        relay._processes.discard(process)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = False

        if port:
            self.server = Server((self.host, port), Handler)
        else:
            port_count = _DYNAMIC_RELAY_PORT_END - _DYNAMIC_RELAY_PORT_START + 1
            first = random.SystemRandom().randrange(port_count)
            for offset in range(port_count):
                candidate = _DYNAMIC_RELAY_PORT_START + (first + offset) % port_count
                try:
                    self.server = Server((self.host, candidate), Handler)
                    break
                except OSError as exc:
                    if exc.errno != errno.EADDRINUSE:
                        raise
            else:
                raise OSError(
                    errno.EADDRINUSE,
                    "no available port in the SEAD relay range "
                    f"{_DYNAMIC_RELAY_PORT_START}-{_DYNAMIC_RELAY_PORT_END}",
                )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        with self._lock:
            for process in self._processes:
                if process.poll() is None:
                    process.kill()
        self.thread.join(timeout=5)


def install_openhands_lease_network(
    network: str, labels: dict[str, str], *, check=None
) -> list[DockerTCPRelay]:
    """Install once in a replay subprocess, before OpenHands creates its runtime."""
    from openhands.runtime.impl.docker.docker_runtime import DockerRuntime

    original = DockerRuntime.init_container
    relays: list[DockerTCPRelay] = []

    def start(runtime, *args, **kwargs):
        if check:
            check()
        original(runtime, *args, **kwargs)
        if check:
            check()
        attrs = runtime.docker_client.containers.get(runtime.container_name).attrs
        if set(attrs["NetworkSettings"]["Networks"]) != {network}:
            raise RuntimeError("leased Target attached to an unexpected network")
        if any(attrs["Config"]["Labels"].get(k) != v for k, v in labels.items()):
            raise RuntimeError("leased Target ownership labels missing")
        relays.append(
            DockerTCPRelay(
                runtime.container_name,
                runtime._container_port,
                port=runtime._host_port,
                python="python3",
            )
        )

    DockerRuntime.init_container = start
    return relays
