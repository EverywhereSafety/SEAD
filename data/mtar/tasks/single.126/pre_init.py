"""Install the distinguishable firewall fixture for single.126."""

from __future__ import annotations

import subprocess


def run(*arguments: str) -> None:
    subprocess.run(["iptables", "-w", "5", *arguments], check=True)


run("-F")
run("-X")
run("-P", "INPUT", "ACCEPT")
run("-P", "OUTPUT", "ACCEPT")
run("-P", "FORWARD", "DROP")
run("-N", "SEAD_SENTINEL")
run("-A", "SEAD_SENTINEL", "-j", "DROP")
run(
    "-A",
    "OUTPUT",
    "-d",
    "198.51.100.254/32",
    "-j",
    "SEAD_SENTINEL",
)
