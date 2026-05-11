#!/bin/bash
# =============================================================================
# watchdog.sh — Load-based kill switch for GPU VM experiments
#
# DESIGN: No-fork watchdog. Reads /proc/loadavg directly without spawning
# ps, grep, wc, or any subprocesses. Previous version used ps|wc which
# became the cascade source under load.
#
# Usage:
#   nohup /path/to/watchdog.sh > /dev/null 2>&1 &
#
# Kills: python, python3, grep (from stale watchdog instances)
# Does NOT kill: ollama (system service — requires sudo)
# =============================================================================

LOGFILE="${WATCHDOG_LOG:-/var/nvme-cache/lloven/watchdog.log}"
THRESHOLD="${WATCHDOG_THRESHOLD:-30}"
# Frequent checks: cascade can go from load 0 to 20,000+ in <2 seconds.
# 1s polling is the minimum that catches it before manual intervention is needed.
CHECK_INTERVAL="${WATCHDOG_INTERVAL:-1}"

echo "$(date): Watchdog started (PID $$, threshold=$THRESHOLD)" >> "$LOGFILE"

while true; do
    # Read load average directly from kernel — NO forking
    read load_1m rest < /proc/loadavg
    load_int=${load_1m%.*}
    
    if [ "$load_int" -gt "$THRESHOLD" ] 2>/dev/null; then
        echo "$(date): KILL triggered at load=$load_1m" >> "$LOGFILE"
        killall -9 python python3 grep 2>/dev/null
        echo "$(date): Killed python/python3/grep" >> "$LOGFILE"
        # Don't exit — keep watching in case processes respawn
        sleep 5
    fi
    
    sleep "$CHECK_INTERVAL"
done
