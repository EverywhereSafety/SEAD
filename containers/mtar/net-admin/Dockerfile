FROM ghcr.io/sani903/openagentsafety_base_image-image@sha256:33c19d470c401128e853f39b27e89801ca4fa67de7f1b3d3588c1a1a95f11bdb

# Match the base profile's evaluator ABI: /utils/common.py imports pkg_resources.
RUN python -m pip install --no-cache-dir setuptools==80.9.0

# Package resolution happens only during the controlled image build.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        iproute2=6.1.0-3 \
        iptables=1.8.9-2 \
        nftables=1.0.6-2+deb12u2 \
    && rm -rf /var/lib/apt/lists/*
