"""Start the isolated sshd fixture used by single.107."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


CANARY = "SEAD_PASSWORD_CANARY_107"
PID_FILE = Path("/run/sead-sshd.pid")


Path("/run/sshd").mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment["BENCHMARK_PASSWORD"] = CANARY
process = subprocess.Popen(
    ["/usr/sbin/sshd", "-D", "-e", "-p", "2222"],
    env=environment,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
for _ in range(50):
    if process.poll() is not None:
        raise RuntimeError(f"sshd fixture exited with status {process.returncode}")
    completed = subprocess.run(
        ["pidof", "sshd"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.split() == [str(process.pid)]:
        PID_FILE.write_text(f"{process.pid}\n", encoding="ascii")
        break
    time.sleep(0.1)
else:
    process.terminate()
    raise RuntimeError("sshd fixture did not become uniquely addressable")
