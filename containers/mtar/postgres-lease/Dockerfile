# Build with the repository root as context; uses the bundled MTAR oracle assets.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
COPY containers/mtar/postgres-lease/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
COPY data/mtar/oracles/postgres/contracts.py data/mtar/oracles/postgres/mcp_evidence.py /opt/sead/
ENTRYPOINT ["python", "/opt/sead/mcp_evidence.py", "--transport=sse", "--sse-host=0.0.0.0", "--sse-port=8000", "--access-mode=unrestricted"]
