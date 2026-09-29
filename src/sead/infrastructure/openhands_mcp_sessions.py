"""Keep OpenHands MCP connections alive for stateful replay tools.

OpenHands 0.54 creates a new HTTP MCP client for every tool action.  That is
fine for stateless servers, but a browser page belongs to its MCP session, so
Playwright navigation/login state otherwise disappears between actions.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import MethodType
from typing import Any


_MCP_SESSION_HEARTBEAT_SECONDS = 20.0


def call_on_runtime_mcp_loop(
    runtime: Any, action: Any, *, timeout: float = 30, call_tool_mcp: Any = None,
) -> Any:
    """Read the replay's MCP session on its owning loop, live or paused.

    EventStream runs each runtime callback on a persistent, single-thread pool.
    Its asyncio loop is idle between actions, so run_coroutine_threadsafe alone
    cannot wake it. Submit to that same pool without emitting Target events.
    """
    dispatch = call_tool_mcp or runtime.call_tool_mcp

    async def invoke():
        return await asyncio.wait_for(dispatch(action), timeout=timeout)

    loop = getattr(runtime, "_sead_mcp_loop", None)
    if loop is not None and loop.is_running():
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            raise RuntimeError("synchronous investigation must run outside the MCP owner loop")
        future = asyncio.run_coroutine_threadsafe(invoke(), loop)
        try:
            return future.result(timeout=timeout + 5)
        except TimeoutError:
            future.cancel()
            raise

    from openhands.events.stream import EventStreamSubscriber

    stream = runtime.event_stream
    pool = stream._thread_pools[EventStreamSubscriber.RUNTIME][runtime.sid]

    def read():
        return asyncio.get_event_loop().run_until_complete(
            invoke()
        )

    future = pool.submit(read)
    try:
        return future.result(timeout=timeout + 5)
    except TimeoutError:
        future.cancel()
        raise


def install_persistent_mcp_sessions(
    runtime: Any, *, timeout_seconds: float = 300
) -> None:
    """Cache MCP clients on one replay runtime and keep their transports open."""

    async def connect_clients(self: Any) -> list[Any]:
        from openhands.mcp.utils import create_mcp_clients

        config = self.get_mcp_config()
        clients = await create_mcp_clients(
            config.sse_servers,
            config.shttp_servers,
            self.sid,
        )
        for client in clients:
            if client.client is None:
                raise RuntimeError("OpenHands created an MCP client without a transport")
            client.client._session_kwargs["read_timeout_seconds"] = timedelta(
                seconds=timeout_seconds
            )
            # MCPClient.connect_http() only opens a temporary connection to
            # list tools. Enter once more and retain this session for every
            # subsequent action in the replay.
            await client.client.__aenter__()
        self._sead_mcp_clients = clients
        self._sead_mcp_heartbeat_task = asyncio.create_task(
            keep_clients_alive(self, clients)
        )
        return clients

    async def keep_clients_alive(self: Any, clients: list[Any]) -> None:
        """Keep stateful Streamable HTTP sessions alive during model thinking."""

        try:
            while getattr(self, "_sead_mcp_clients", None) is clients:
                await asyncio.sleep(_MCP_SESSION_HEARTBEAT_SECONDS)
                for client in clients:
                    transport = getattr(client, "client", None)
                    if transport is not None:
                        await transport.ping()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The foreground action path performs the authoritative reconnect
            # and records any tool outcome. A heartbeat failure is only a hint
            # that the cached transport may need that fallback.
            return

    async def reconnect_clients(self: Any, clients: list[Any]) -> list[Any]:
        heartbeat = getattr(self, "_sead_mcp_heartbeat_task", None)
        if heartbeat is not None:
            heartbeat.cancel()
        for client in clients:
            transport = getattr(client, "client", None)
            if transport is None:
                continue
            try:
                await transport.__aexit__(None, None, None)
            except Exception:
                # The transport is already invalid when the server reports a
                # terminated session. Best-effort close must not prevent a
                # fresh Streamable HTTP session from being established.
                pass
        self._sead_mcp_clients = None
        return await connect_clients(self)

    def matching_client(clients: list[Any], tool_name: str) -> Any:
        return next(
            (
                client
                for client in clients
                if tool_name in {tool.name for tool in client.tools}
            ),
            None,
        )

    async def call_tool_mcp(self: Any, action: Any) -> Any:
        self._sead_mcp_loop = asyncio.get_running_loop()
        from mcp import McpError

        from openhands.events.observation.mcp import MCPObservation

        clients = getattr(self, "_sead_mcp_clients", None)
        if clients is None:
            clients = await connect_clients(self)

        matching = matching_client(clients, action.name)
        if matching is None or matching.client is None:
            raise ValueError(f"No matching MCP agent found for tool name: {action.name}")

        try:
            response = await matching.client.call_tool_mcp(
                name=action.name,
                arguments=action.arguments,
            )
            content = json.dumps(response.model_dump(mode="json"))
        except McpError as exc:
            if "session terminated" not in str(exc).casefold():
                content = json.dumps(
                    {"isError": True, "error": str(exc), "content": []}
                )
            else:
                clients = await reconnect_clients(self, clients)
                matching = matching_client(clients, action.name)
                if matching is None or matching.client is None:
                    raise ValueError(
                        f"No matching MCP agent found for tool name: {action.name}"
                    )
                try:
                    response = await matching.client.call_tool_mcp(
                        name=action.name,
                        arguments=action.arguments,
                    )
                    content = json.dumps(response.model_dump(mode="json"))
                except McpError as retry_exc:
                    content = json.dumps(
                        {
                            "isError": True,
                            "error": str(retry_exc),
                            "content": [],
                        }
                    )
        return MCPObservation(
            content=content,
            name=action.name,
            arguments=action.arguments,
        )

    runtime.call_tool_mcp = MethodType(call_tool_mcp, runtime)
