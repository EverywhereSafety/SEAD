#!/bin/bash
set -euo pipefail

sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config && systemctl reload sshd
