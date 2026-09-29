FROM ghcr.io/sani903/openagentsafety_base_image-image@sha256:33c19d470c401128e853f39b27e89801ca4fa67de7f1b3d3588c1a1a95f11bdb

# Match the base profile's evaluator ABI: /utils/common.py imports pkg_resources.
RUN python -m pip install --no-cache-dir setuptools==80.9.0

# Package resolution happens only during the controlled image build.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        binutils=2.40-2 \
        gdb=13.1-3 \
        openssh-server=1:9.2p1-2+deb12u10 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd
