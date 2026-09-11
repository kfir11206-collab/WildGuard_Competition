#!/bin/bash
for f in /sys/devices/system/cpu/cpu[0-5]/cpufreq/scaling_max_freq; do
  echo 1344000 | sudo tee "$f" >/dev/null
done
echo 918000000 | sudo tee /sys/class/devfreq/17000000.gpu/max_freq >/dev/null
echo "cpu caps: $(sort -u /sys/devices/system/cpu/cpu[0-5]/cpufreq/scaling_max_freq | xargs)   (want 1344000)"
echo "gpu cap:  $(cat /sys/class/devfreq/17000000.gpu/max_freq)   (want 918000000)"
echo "memory:   $(cat /sys/class/devfreq/bwmgr/max_freq)   (want 3199000000)"
