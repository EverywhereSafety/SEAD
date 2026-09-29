"""Default and task-specific routing to exclusive environment pools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

from sead.config.loader import read_yaml, resolve_paths

SERVICES = frozenset({"reddit", "postgres", "gitlab", "owncloud"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def environment_config(execution):
    value = execution.get("environments")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {
        "mode", "registry", "registry_digest", "acquire_timeout_seconds", "rpc_timeout_seconds"
    }:
        raise ValueError("invalid execution.environments")
    if value.get("mode") != "leased" or not value.get("registry"):
        raise ValueError("execution.environments requires mode: leased and registry")
    if any(key in execution for key in ("postgres", "forum_pool", "tac_pool_config")):
        raise ValueError("leased environments conflict with legacy postgres/forum_pool/tac_pool_config")
    for key in ("acquire_timeout_seconds", "rpc_timeout_seconds"):
        if key in value and (isinstance(value[key], bool) or not 0 < float(value[key]) < float("inf")):
            raise ValueError(f"invalid environments.{key}")
    return dict(value)


def required_service(dependencies):
    dependencies = set(dependencies)
    selected = dependencies & SERVICES
    if "mcp-postgres" in dependencies:
        selected.add("postgres")
    unsupported = dependencies - SERVICES - {"mcp-postgres", "mcp-playwright", "mcp-filesystem"}
    if unsupported:
        raise ValueError(f"unsupported leased services: {sorted(unsupported)}")
    if len(selected) > 1:
        raise ValueError("UNSUPPORTED_ENVIRONMENT_BUNDLE: multiple mutable services")
    if "mcp-playwright" in dependencies and not selected:
        raise ValueError("leased browser task must declare its service")
    return next(iter(selected), None)


def load_pool(path):
    path = Path(path).absolute()
    value = resolve_paths(read_yaml(path), path.parent)
    allowed = {"schema_version", "pool_id", "service", "protocol", "provider", "state_directory",
               "capacity", "max_parallel_creates", "ttl_seconds", "heartbeat_interval_seconds",
               "startup_timeout_seconds", "cleanup_timeout_seconds", "poll_interval_seconds",
               "profiles", "allowed_profiles", "manager_socket", "provider_config"}
    allowed.add("request_timeout_seconds")
    if value.get("schema_version") != "sead-environment-pool-v1" or set(value) - allowed:
        raise ValueError(f"invalid environment pool: {path}")
    if value.get("service") not in SERVICES or value.get("provider") != value["service"]:
        raise ValueError("invalid pool provider/service")
    if not value.get("pool_id"):
        raise ValueError("pool_id is required")
    protocol = value.get("protocol", "environment-lease-v2")
    if protocol not in {"environment-lease-v2", "postgres-v1"}:
        raise ValueError("unknown environment lease protocol")
    if protocol == "postgres-v1" and value["service"] != "postgres":
        raise ValueError("postgres-v1 requires postgres provider")
    for key in ("capacity", "max_parallel_creates"):
        number = value.get(key, 1)
        if type(number) is not int or number < 1:
            raise ValueError(f"pool.{key} must be a positive integer")
    for key in ("ttl_seconds", "heartbeat_interval_seconds", "startup_timeout_seconds",
                "cleanup_timeout_seconds", "poll_interval_seconds", "request_timeout_seconds"):
        if key in value and (isinstance(value[key], bool) or not 0 < float(value[key]) < float("inf")):
            raise ValueError(f"invalid pool.{key}")
    if value.get("heartbeat_interval_seconds", 10) >= value.get("ttl_seconds", 180) / 2:
        raise ValueError("heartbeat interval must be less than half the TTL")
    if protocol == "postgres-v1":
        if not value.get("manager_socket"):
            raise ValueError("postgres-v1 requires manager_socket")
    else:
        if not value.get("state_directory"):
            raise ValueError("pool state_directory is required")
        value["manager_socket"] = str(Path(value["state_directory"]) / "manager.sock")
    if len(value["manager_socket"].encode()) >= 108:
        raise ValueError("manager socket path exceeds Linux AF_UNIX limit; shorten state_directory")
    profile_path = Path(value["profiles"])
    profiles = resolve_paths(read_yaml(profile_path), profile_path.parent)
    if set(profiles) != {"schema_version", "profiles"} or profiles["schema_version"] != "sead-environment-profiles-v1":
        raise ValueError("invalid environment profiles")
    selected = value.get("allowed_profiles", list(profiles["profiles"]))
    value["profile_definitions"] = {}
    for name in selected:
        profile = dict(profiles["profiles"][name])
        if profile.get("service") != value["service"]:
            raise ValueError("profile/service mismatch")
        value["profile_definitions"][name] = profile
    value["config_digest"] = digest(value)
    return value


class EnvironmentRegistry:
    def __init__(self, path):
        path = Path(path).absolute()
        value = read_yaml(path)
        version = value.get("schema_version")
        expected_keys = (
            {"schema_version", "pools", "routing"}
            if version == "sead-environment-registry-v1"
            else {"schema_version", "pools", "routing", "task_routing"}
            if version == "sead-environment-registry-v2"
            else {"schema_version", "benchmark", "pools", "routing", "task_routing"}
        )
        if version not in {
            "sead-environment-registry-v1",
            "sead-environment-registry-v2",
            "sead-environment-registry-v3",
        } or set(value) != expected_keys:
            raise ValueError("invalid environment registry")
        self.benchmark = value.get("benchmark")
        if self.benchmark is not None:
            if (
                not isinstance(self.benchmark, dict)
                or set(self.benchmark) != {
                    "kind", "collection", "group", "expected_task_count"
                }
                or self.benchmark["kind"] != "mtar"
                or self.benchmark["collection"] != "ready187-v1"
                or self.benchmark["group"]
                not in {"official75", "expansion112", "all187"}
                or type(self.benchmark["expected_task_count"]) is not int
                or self.benchmark["expected_task_count"] < 1
            ):
                raise ValueError("invalid environment registry benchmark binding")
        self.pools = {}
        self.pool_paths = {}
        for key, row in value["pools"].items():
            if set(row) != {"config"}:
                raise ValueError("pool reference requires only config")
            pool_path = (path.parent / row["config"]).absolute()
            self.pool_paths[key] = pool_path
            self.pools[key] = load_pool(pool_path)
        self.task_routing = value.get("task_routing", {})
        self.routing = value["routing"]
        for service, route in self.routing.items():
            self._validate_route(service, route)
            self.resolve(service)
        if not isinstance(self.task_routing, dict):
            raise ValueError("task_routing must be a mapping")
        for task_id, routes in self.task_routing.items():
            if not isinstance(task_id, str) or not task_id or not isinstance(routes, dict) or not routes:
                raise ValueError("invalid task routing")
            for service, route in routes.items():
                self._validate_route(service, route)
                self.resolve(service, task_id)
        digest_input = {
            "pools": {key: pool["config_digest"] for key, pool in self.pools.items()},
            "routing": self.routing,
        }
        if version in {
            "sead-environment-registry-v2",
            "sead-environment-registry-v3",
        }:
            digest_input["task_routing"] = self.task_routing
        if self.benchmark is not None:
            digest_input["benchmark"] = self.benchmark
        self.digest = digest(digest_input)

    def validate_catalog(self, catalog):
        """Fail closed when a collection run uses a registry for another group."""

        if self.benchmark is None:
            return
        resolution = getattr(catalog, "mtar_collection", None)
        if resolution is None:
            return  # Legacy single-root configs cannot declare a collection group.
        expected = self.benchmark
        if (
            getattr(catalog, "kind", None) != expected["kind"]
            or resolution.name != expected["collection"]
            or resolution.group != expected["group"]
            or len(resolution.tasks) != expected["expected_task_count"]
        ):
            raise ValueError(
                "environment registry benchmark binding does not match "
                f"{resolution.name}/{resolution.group}"
            )

    @staticmethod
    def _validate_route(service, route):
        if service not in SERVICES or not isinstance(route, dict) or set(route) != {"pool", "profile"}:
            raise ValueError("invalid service routing")

    def resolve(self, service, task_id=None):
        try:
            route = self.task_routing.get(task_id, {}).get(service)
            if route is None:
                route = self.routing[service]
            pool = self.pools[route["pool"]]
            profile = pool["profile_definitions"][route["profile"]]
        except KeyError as exc:
            raise ValueError(f"no environment pool/profile for {service}") from exc
        if pool["service"] != service:
            raise ValueError("pool route/service mismatch")
        return pool, route["profile"], profile

    def plan(self, dependencies, task_id=None):
        service = required_service(dependencies)
        if service is None:
            return None
        pool, profile_id, profile = self.resolve(service, task_id)
        return {"service": service, "pool_id": pool["pool_id"], "profile_id": profile_id,
                "profile_digest": digest(profile), "capacity": pool["capacity"]}


def public_deployments(execution, dependencies):
    config = environment_config(execution)
    if not config:
        return None
    service = required_service(dependencies)
    if service is None or service == "postgres":
        return {}
    _, _, profile = EnvironmentRegistry(config["registry"]).resolve(service)
    return {service: {"display_name": service, "url": profile["public_url"],
                      "credentials": profile["credentials"],
                      "operational_notes": profile.get("operational_notes", [])}}
