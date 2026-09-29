"""Per-run admission limits; the environment manager owns physical capacity."""

from sead.environments.registry import EnvironmentRegistry, environment_config


def leased_resource_plan(catalog, task_ids, execution, suite):
    config = environment_config(execution)
    if not config:
        return None
    registry = EnvironmentRegistry(config["registry"])
    registry.validate_catalog(catalog)
    limits = suite.get("service_task_limits", {})
    if not isinstance(limits, dict) or set(limits) - set(registry.routing):
        raise ValueError("unknown service_task_limits")
    capacities = {}
    for service in registry.routing:
        limit = limits.get(service, registry.resolve(service)[0]["capacity"])
        if type(limit) is not int or limit < 1:
            raise ValueError("service task limits must be positive integers")
        capacities[f"service:{service}"] = limit
    family_limits = suite.get("tool_family_task_limits", {})
    if not isinstance(family_limits, dict) or set(family_limits) - {
        "filesystem", "terminal"
    }:
        raise ValueError("unknown tool_family_task_limits")
    for family, limit in family_limits.items():
        if type(limit) is not int or limit < 1:
            raise ValueError("tool family task limits must be positive integers")
        capacities[f"family:{family}"] = limit
    web_limit = suite.get("max_concurrent_web_tasks")
    if web_limit is not None:
        if type(web_limit) is not int or web_limit < 1:
            raise ValueError("max_concurrent_web_tasks must be positive or null")
        capacities["capacity:web"] = web_limit
    resources = {}
    for task_id in task_ids:
        dependencies = catalog.dependencies(task_id)
        plan = registry.plan(dependencies, task_id)
        keys = [f"service:{plan['service']}"] if plan else []
        if plan:
            pool_key = f"pool:{plan['pool_id']}"
            configured = capacities.setdefault(pool_key, plan["capacity"])
            if configured != plan["capacity"]:
                raise ValueError("environment pool has inconsistent capacities")
            keys.append(pool_key)
        elif "mcp-filesystem" in dependencies and "filesystem" in family_limits:
            keys.append("family:filesystem")
        elif "terminal" in family_limits:
            keys.append("family:terminal")
        if plan and plan["service"] != "postgres" and web_limit is not None:
            keys.append("capacity:web")
        resources[task_id] = tuple(keys)
    return resources, capacities
