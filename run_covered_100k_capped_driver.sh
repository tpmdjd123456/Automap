#!/bin/bash
# Covered-100k run with the scoring-only pair cap, honoring the dama advisor's
# resource policy (see memory feedback_dama-resource-policy):
#   - low-resource rate-test on a small cover-heavy slice BEFORE the full run
#   - a memory watchdog that hard-aborts the run before RAM saturation
# Runs server-side under setsid so it survives SSH drops.
set -u
cd /home/automap/Automap || exit 1
PY=.venv/bin/python
export AUTOMAP_SCORE_PAIR_CAP=150

# ---- tunables ----
CORPUS=data/wdc_covered_100k.jsonl
OUTDIR=output/wdc_covered_100k_capped
LOG=output/wdc_covered_100k_capped.log
WATCHLOG=output/wdc_covered_100k_capped_watchdog.log
TEST_N=15000                 # cover-heavy slice (cover tables are written first)
TEST_TIMEOUT=600             # rate-test must finish within 10 min
MEM_FLOOR_GB=15              # abort full run if system MemAvailable drops below this
RSS_CAP_GB=120               # abort full run if OUR total RSS exceeds this

log() { echo "[driver $(date '+%F %T')] $*"; }

# make sure any abandoned runs are gone
pkill -f "run_covered_250k_driver" 2>/dev/null
pkill -f "main.py.*wdc_covered_250k" 2>/dev/null
sleep 2

# ---- 0. coverage sanity ----
log "re-verifying covered-100k coverage (expect 30134)"
COV=$($PY -u extract_covered_pairs.py data/benchmark-web.txt "$CORPUS" /tmp/cov_100k.txt 2>&1 \
        | grep "Covered gold pairs" | grep -oE "[0-9]+$")
log "COVERAGE=$COV"
if [ "$COV" != "30134" ]; then
    log "COVERAGE MISMATCH ($COV != 30134) — ABORTING"
    exit 1
fi

# ---- 1. low-resource rate-test (advisor rule 4) ----
# First TEST_N tables are the value-rich cover tables, the densest / most
# expensive region for Stage 6. If the pair cap makes THIS finish quickly, the
# full (more diluted) run certainly will. `timeout` is the pass/fail gate.
log "RATE-TEST: pipeline on first ${TEST_N} (cover-heavy) tables, cap=${AUTOMAP_SCORE_PAIR_CAP}, timeout=${TEST_TIMEOUT}s"
head -n "$TEST_N" "$CORPUS" > /tmp/covtest_${TEST_N}.jsonl
mkdir -p "${OUTDIR}_test"
timeout "$TEST_TIMEOUT" $PY -u main.py \
    --corpus_path /tmp/covtest_${TEST_N}.jsonl \
    --output_folder "${OUTDIR}_test/" \
    --threshold 0.3 --theta 0.95 --parallel_workers 8 \
    --max_bucket_size 250 --no_save_index --string_matcher jaccard \
    > "${LOG}.ratetest" 2>&1
RC=$?
if [ $RC -eq 124 ]; then
    log "RATE-TEST TIMED OUT after ${TEST_TIMEOUT}s — pair cap not fast enough on the dense region. ABORTING full run."
    log "  inspect ${LOG}.ratetest ; consider lowering AUTOMAP_SCORE_PAIR_CAP or max_bucket_size"
    exit 2
elif [ $RC -ne 0 ]; then
    log "RATE-TEST FAILED (exit $RC) — see ${LOG}.ratetest. ABORTING."
    exit $RC
fi
log "RATE-TEST PASSED (finished in <${TEST_TIMEOUT}s). Stage-6 timing:"
grep -aE "^\[Stage 6|Time:|Non-zero positive edges|Total time" "${LOG}.ratetest" | tail -4

# ---- 2. memory watchdog (advisor rules 2 & 1) ----
# Hard switch-off: kills the full run if the box is about to saturate RAM or our
# own RSS runs away. Logs the reason. Backgrounded; stops when the run ends.
mem_watchdog() {
    while pgrep -f "main.py.*wdc_covered_100k.jsonl" >/dev/null 2>&1; do
        avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        pids=$(pgrep -f "main.py.*wdc_covered_100k.jsonl")
        ourrss=$(ps -o rss= -p $pids 2>/dev/null | awk '{s+=$1} END{print int(s/1048576)}')
        echo "[watchdog $(date '+%T')] MemAvailable=${avail}GB ourRSS=${ourrss}GB" >> "$WATCHLOG"
        if [ "${avail:-999}" -lt "$MEM_FLOOR_GB" ]; then
            echo "[watchdog $(date '+%T')] MemAvailable ${avail}GB < ${MEM_FLOOR_GB}GB floor — KILLING run to protect the box" >> "$WATCHLOG"
            pkill -TERM -f "main.py.*wdc_covered_100k.jsonl"; sleep 5
            pkill -9 -f "main.py.*wdc_covered_100k.jsonl" 2>/dev/null
            break
        fi
        if [ "${ourrss:-0}" -gt "$RSS_CAP_GB" ]; then
            echo "[watchdog $(date '+%T')] our RSS ${ourrss}GB > ${RSS_CAP_GB}GB cap — KILLING run (possible leak)" >> "$WATCHLOG"
            pkill -TERM -f "main.py.*wdc_covered_100k.jsonl"; sleep 5
            pkill -9 -f "main.py.*wdc_covered_100k.jsonl" 2>/dev/null
            break
        fi
        sleep 30
    done
    echo "[watchdog $(date '+%T')] run ended; watchdog exiting" >> "$WATCHLOG"
}

# ---- 3. full covered-100k run under the watchdog ----
log "launching FULL covered-100k pipeline (cap=${AUTOMAP_SCORE_PAIR_CAP}, mem floor=${MEM_FLOOR_GB}GB, rss cap=${RSS_CAP_GB}GB)"
: > "$WATCHLOG"
mkdir -p "$OUTDIR"
$PY -u main.py \
    --corpus_path "$CORPUS" \
    --output_folder "${OUTDIR}/" \
    --threshold 0.3 --theta 0.95 --parallel_workers 8 \
    --max_bucket_size 250 --no_save_index --string_matcher jaccard \
    > "$LOG" 2>&1 &
MAIN=$!
mem_watchdog &
WATCH=$!
wait $MAIN
RC=$?
kill "$WATCH" 2>/dev/null
log "pipeline exited code $RC — DONE (watchdog log: $WATCHLOG)"
