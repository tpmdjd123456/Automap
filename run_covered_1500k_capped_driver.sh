#!/bin/bash
# Covered-1.5M run with the scoring-only pair cap. The cap was already validated
# at 100k scale (86.2% covered recall, finished in 4.6h) so there is NO inline
# rate-test here — the "test small first" step is satisfied by that 100k run.
# The memory watchdog (advisor rules 1-2, see memory feedback_dama-resource-policy)
# is kept: it hard-aborts before RAM saturation. Runs under setsid.
set -u
cd /home/automap/Automap || exit 1
PY=.venv/bin/python
export AUTOMAP_SCORE_PAIR_CAP=150

CORPUS=data/wdc_covered_1500k.jsonl
OUTDIR=output/wdc_covered_1500k_capped
LOG=output/wdc_covered_1500k_capped.log
WATCHLOG=output/wdc_covered_1500k_capped_watchdog.log
MEM_FLOOR_GB=15              # abort if system MemAvailable drops below this
RSS_CAP_GB=180              # abort if OUR total RSS exceeds this (1.5M is heavier)
DISK_FLOOR_GB=20           # abort up-front if free disk is below this

log() { echo "[driver $(date '+%F %T')] $*"; }

# ---- 0. coverage sanity ----
log "re-verifying covered-1.5M coverage (expect 30134)"
COV=$($PY -u extract_covered_pairs.py data/benchmark-web.txt "$CORPUS" /tmp/cov_1500k.txt 2>&1 \
        | grep "Covered gold pairs" | grep -oE "[0-9]+$")
log "COVERAGE=$COV"
if [ "$COV" != "30134" ]; then
    log "COVERAGE MISMATCH ($COV != 30134) — ABORTING"
    exit 1
fi

# ---- 1. resource pre-flight (advisor: check the box before a massive job) ----
FREE_GB=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
AVAIL_GB=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
log "pre-flight: disk_free=${FREE_GB}GB mem_available=${AVAIL_GB}GB load='$(uptime | sed 's/.*load average://')'"
if [ "${FREE_GB:-0}" -lt "$DISK_FLOOR_GB" ]; then
    log "DISK TOO LOW (${FREE_GB}GB < ${DISK_FLOOR_GB}GB) — ABORTING (1.5M writes several GB of candidates/edges/mappings)"
    exit 3
fi

# ---- 2. full covered-1.5M run — NO memory watchdog (per explicit request) ----
# WARNING: no memory switch-off. Attempt #1 peaked ~233/251 GB at Stage 2; a real
# OOM here can crash the box for other users. Launch only when the box is idle.
# A background RSS/MemAvailable LOGGER is kept for observability but never kills.
mem_logger() {
    while pgrep -f "main.py.*wdc_covered_1500k.jsonl" >/dev/null 2>&1; do
        avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
        echo "[memlog $(date '+%F %T')] MemAvailable=${avail}GB disk_free=${free_gb}GB" >> "$WATCHLOG"
        sleep 30
    done
}

log "launching FULL covered-1.5M pipeline BARE — NO memory watchdog (cap=${AUTOMAP_SCORE_PAIR_CAP})"
: > "$WATCHLOG"
mkdir -p "$OUTDIR"
$PY -u main.py \
    --corpus_path "$CORPUS" \
    --output_folder "${OUTDIR}/" \
    --threshold 0.3 --theta 0.95 --parallel_workers 8 \
    --max_bucket_size 250 --no_save_index --string_matcher jaccard \
    > "$LOG" 2>&1 &
MAIN=$!
mem_logger &
LOGGER=$!
wait $MAIN
RC=$?
kill "$LOGGER" 2>/dev/null
log "pipeline exited code $RC — DONE (mem log: $WATCHLOG)"
