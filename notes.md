# Project Notes
There is nothing particularly important in here. I'm basically treating it like a piece of scratch paper that I can't accidentally throw away.

## Raspberry Pi 3
* Architecture: 64-bit ARMv8 

## Docker
* The Raspbian Lite OS does not enable cgroups for memory by default, which means Docker containers cannot have specific memory allocation limits set. To fix this, add `cgroup_enable=memory cgroup_memory=1 swapaccount=1` to the end of the single line in /boot/cmdline.txt, then reboot. Use the command `docker info` to check that it worked.
* You may need to increase the size of the swapfile on Raspberry Pi devices to prevent OOM ("Out Of Memory") errors from killing your containers. To do this, stop swapping temporarily with the `sudo dphys-swapfile swapoff` command, then edit the `/etc/dphys-swapfile` file to set CONF_SWAPSIZE to the preferred size in megabytes. Then use `sudo dphys-swapfile setup` to create the new swapfile, and `sudo dphys-swapfile swapon` to start using it again. Reboot required to allow access to the new swapfile for running processes.
