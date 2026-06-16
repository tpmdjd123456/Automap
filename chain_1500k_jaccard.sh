#!/bin/bash
# Vertica extract -> 1.5M Jaccard pipeline. If extract fails, pipeline skipped.
set -u
cd ~/Automap

CORPUS=data/vertica_filtered_1500k.jsonl
OUTDIR=output/results_1500k_jaccard
EXTRACT_LOG=extract_1500k_retry.log
PIPELINE_LOG=run_paper_v1500k_jaccard.log

echo "=== START $(date) ===" > chain_1500k.log

echo ""                                  >> chain_1500k.log
echo "--- Phase 1: Vertica extract ---"  >> chain_1500k.log
echo "start: $(date)"                    >> chain_1500k.log
.venv/bin/python -u extract_filtered_sample.py \
    --n 1500000 --out "$CORPUS" \
    > "$EXTRACT_LOG" 2>&1
EXTRACT_EXIT=$?
echo "end:   $(date)   exit_code=$EXTRACT_EXIT" >> chain_1500k.log

if [ $EXTRACT_EXIT -ne 0 ]; then
    echo "ABORT: extract failed; not running pipeline" >> chain_1500k.log
    exit 1
fi
if [ ! -s "$CORPUS" ]; then
    echo "ABORT: extract exit=0 but corpus file is empty/missing" >> chain_1500k.log
    exit 1
fi

echo ""                                       >> chain_1500k.log
echo "--- Phase 2: 1.5M Jaccard pipeline ---" >> chain_1500k.log
echo "start: $(date)"                         >> chain_1500k.log
mkdir -p "$OUTDIR"
.venv/bin/python -u main.py \
    --corpus_path "$CORPUS" \
    --output_folder "$OUTDIR/" \
    --threshold 0.3 --theta 0.95 \
    --parallel_workers 8 \
    --max_bucket_size 250 \
    --no_save_index \
    --string_matcher jaccard \
    > "$PIPELINE_LOG" 2>&1
PIPELINE_EXIT=$?
echo "end:   $(date)   exit_code=$PIPELINE_EXIT" >> chain_1500k.log

echo ""                  >> chain_1500k.log
echo "=== END $(date) ===" >> chain_1500k.log
