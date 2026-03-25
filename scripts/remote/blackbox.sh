#!/bin/bash
# =============================================================================
# blackbox.sh - Passive diagnostic recorder for GPU VM cascades
#
# No-fork design: reads /proc directly. Only nvidia-smi forks (every 10s).
# Output: JSONL to /var/nvme-cache/lloven/blackbox.jsonl (max 1000 lines)
# =============================================================================

LOGFILE="/var/nvme-cache/lloven/blackbox.jsonl"
MAX_SAMPLES=1000
CHECK_INTERVAL=2
GPU_INTERVAL=5

mkdir -p "$(dirname "$LOGFILE")" 2>/dev/null

gpu_mem_mb=0
gpu_util_pct=0
iteration=0

echo "$(date): Blackbox started (PID $$)" >> "${LOGFILE%.jsonl}_status.log"

while true; do
    # 1. Timestamp
    ts=$(date +%Y-%m-%dT%H:%M:%S)

    # 2. Load
    read load_1m load_5m load_15m procs_frac last_pid < /proc/loadavg
    procs_running=${procs_frac%%/*}

    # 3. Procs blocked (from /proc/stat)
    procs_blocked=0
    while IFS= read -r line; do
        case "$line" in
            "procs_blocked "*)
                procs_blocked=${line#procs_blocked }
                break
                ;;
        esac
    done < /proc/stat

    # 4. Python count + ppids
    python_count=0
    python_ppids=""
    for comm_file in /proc/[0-9]*/comm; do
        read -r cmd < "$comm_file" 2>/dev/null || continue
        case "$cmd" in
            python|python3)
                python_count=$((python_count + 1))
                pid_dir="${comm_file%/comm}"
                read -r stat_line < "${pid_dir}/stat" 2>/dev/null || continue
                stat_after="${stat_line##*) }"
                read -r _state ppid _rest <<< "$stat_after"
                if [ -n "$python_ppids" ]; then
                    python_ppids="${python_ppids},${ppid}"
                else
                    python_ppids="${ppid}"
                fi
                ;;
        esac
    done
    ppids_json="[${python_ppids}]"

    # 5. NFS ops (from mountstats)
    nfs_ops=0
    in_nfs=0
    while IFS= read -r line; do
        case "$line" in
            *"/mnt/nfs/home"*) in_nfs=1 ;;
            "device "*) [ $in_nfs -eq 1 ] && break ;;
            *"operations:"*)
                if [ $in_nfs -eq 1 ]; then
                    nfs_ops=${line##*: }
                fi
                ;;
        esac
    done < /proc/self/mountstats 2>/dev/null

    # 6. GPU (low frequency fork)
    iteration=$((iteration + 1))
    if [ $((iteration % GPU_INTERVAL)) -eq 0 ]; then
        gpu_out=$(timeout 3 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$gpu_out" ]; then
            IFS=', ' read -r gpu_mem_mb gpu_util_pct <<< "$gpu_out"
        fi
    fi

    # 7. Write JSON line
    printf '{"ts":"%s","load_1m":%s,"load_5m":%s,"procs_run":%s,"procs_blk":%s,"py":%s,"ppids":%s,"nfs":%s,"gpu_mb":%s,"gpu_pct":%s}\n' \
        "$ts" "$load_1m" "$load_5m" "$procs_running" "$procs_blocked" \
        "$python_count" "$ppids_json" "$nfs_ops" "$gpu_mem_mb" "$gpu_util_pct" \
        >> "$LOGFILE"

    # 8. Ringbuffer (every 100 iterations)
    if [ $((iteration % 100)) -eq 0 ]; then
        lc=$(wc -l < "$LOGFILE" 2>/dev/null)
        if [ "${lc:-0}" -gt "$MAX_SAMPLES" ]; then
            tail -n "$MAX_SAMPLES" "$LOGFILE" > "${LOGFILE}.tmp"
            mv "${LOGFILE}.tmp" "$LOGFILE"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
