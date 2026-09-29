# Local runtime images

The repository provides Dockerfiles instead of distributing locally built images.
Entries for these images in `data/mtar/runtime_profiles.yml` start with
`image: null` and `available: false`. Build the images on your machine, then
configure their actual IDs before using the corresponding runtime profiles.
Upstream repository references remain pinned in the Dockerfiles and registry.
Access to those images and package repositories is required during the build.

## Build contexts and configuration

Run Docker builds from the repository root. For example, the base image uses:

```bash
docker build -t sead/base:local containers/mtar/base
docker image inspect --format '{{.Id}}' sead/base:local
```

Use the build context and Dockerfile listed below for each image. Local tags are
only convenient build names; the runtime registry requires immutable image IDs
or repository references with a digest.

| Image | Dockerfile | Build context | Configuration entry |
| --- | --- | --- | --- |
| Base | `containers/mtar/base/Dockerfile` | `containers/mtar/base` | `profiles.base` and `profiles.playwright-mcp` |
| Filesystem MCP | `containers/mtar/filesystem-mcp/Dockerfile` | `containers/mtar/filesystem-mcp` | `services.filesystem-mcp` |
| Playwright MCP | `containers/mtar/playwright-mcp/Dockerfile` | `containers/mtar/playwright-mcp` | `services.playwright-mcp` |
| Ptrace/SSH | `containers/mtar/ptrace-sshd/Dockerfile` | `containers/mtar/ptrace-sshd` | `profiles.ptrace-sshd` |
| Network administration | `containers/mtar/net-admin/Dockerfile` | `containers/mtar/net-admin` | `profiles.net-admin` |
| PostgreSQL lease MCP | `containers/mtar/postgres-lease/Dockerfile` | Repository root (`.`) | External PostgreSQL lease deployment |

For each successfully built image, update the corresponding entry in
`data/mtar/runtime_profiles.yml`: set `image` to the complete `sha256:...` ID
reported by Docker, `available` to `true`, and `unavailable_reason` to `null`.
Keep these deployment-specific values local; do not commit them to the release
template. The two base-profile entries use the same image. The Playwright MCP
service uses its own image.

The PostgreSQL lease image needs the repository root as its build context because
it copies the bundled oracle files. It is configured through the external lease
deployment, not through a service entry in this registry.

## Evaluator dependencies and external services

The base Dockerfile prepares `/opt/sead/wheelhouse`. The registry defaults
`evaluator_wheelhouse_available` to `false`; enable it only after verifying that
the images used for tasks requiring evaluator packages contain the required
wheelhouse. This flag is shared across runtime profiles. The other Dockerfiles
do not all prepare that directory.

The Reddit service image and its build sources are not included. Its service
entry remains unavailable until a compatible deployment is supplied. The
repository retains upstream GitLab, ownCloud, and OAS image references; their
presence does not provision or verify those services locally.

Unimplemented service and microVM deployment entries are omitted from the
registry. The MTAR task data is retained, including tasks that require those
environments; such tasks cannot run until a compatible runtime profile is added.

Building images alone does not install OpenHands, configure service instances,
or provide the PostgreSQL lease manager. Pinned upstream images and package
versions must still be accessible; a Dockerfile does not guarantee that its
external dependencies will remain available.
