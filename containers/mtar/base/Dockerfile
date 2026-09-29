FROM ghcr.io/sani903/openagentsafety_base_image-image@sha256:33c19d470c401128e853f39b27e89801ca4fa67de7f1b3d3588c1a1a95f11bdb

# OpenAgentSafety's /utils/common.py imports pkg_resources. Recent upstream
# image rebuilds omit setuptools, so pin it as part of the evaluator ABI.
RUN python -m pip install --no-cache-dir setuptools==80.9.0

# Downloaded only while publishing the immutable image.  Evaluator packages
# are installed from this directory after the agent finishes, with --no-index.
RUN mkdir -p /opt/sead/wheelhouse \
    && python -m pip download --only-binary=:all: --dest /opt/sead/wheelhouse \
        psycopg2-binary==2.9.10
