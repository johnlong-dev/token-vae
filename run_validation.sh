#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Token VAE — Full Hypothesis Validation Script
#
# Executes 21 training runs covering reproducibility, data scale, vocab scale,
# ablations, and baselines. Produces validation_results/summary.md.
#
# Usage:
#   ./run_validation.sh                          # full run (~11 hours on M-series Mac)
#   ./run_validation.sh --dry-run                # print commands without executing
#   ./run_validation.sh --section A              # run only section A
#   ./run_validation.sh --skip-existing          # skip runs with existing results.json
#   ./run_validation.sh --section A --dry-run    # preview section A commands
# =============================================================================

RESULTS_DIR="validation_results"
DATA_DIR="data"
TOK_DIR="artifacts/tokenizer"

# --- CLI parsing ---
DRY_RUN=false
SECTION="ALL"
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --section)    SECTION="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Baseline training args (shared across all runs) ---
BASELINE_ARGS=(
    --interp-weight 0.05
    --interp-pairs 64
    --interp-entropy-target 8.0
    --interp-holdout-num 200
    --identity-vocab-repeats 5
    --kl-weight 0.03
    --kl-free-bits 0.25
    --ifw-alpha 1.0
    --prior-reg-weight 0.50
    --prior-reg-interval 1
    --prior-entropy-target 4.0
    --prior-max-freq-target 0.05
    --prior-marginal-entropy-target 7.0
    --prior-hhi-target 0.001
)

# --- Helpers ---

log() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
}

run_cmd() {
    if $DRY_RUN; then
        echo "  [DRY RUN] $*"
    else
        "$@"
    fi
}

train_and_eval() {
    # Usage: train_and_eval <output_dir> <data> <tokenizer> <seed> [--allow-eval-fail] [extra_train_args...]
    local output_dir="$1"; shift
    local data="$1"; shift
    local tokenizer="$1"; shift
    local seed="$1"; shift
    local allow_eval_fail=false
    if [[ $# -gt 0 && "$1" == "--allow-eval-fail" ]]; then
        allow_eval_fail=true
        shift
    fi
    # Remaining args are extra training overrides

    local model_path="${output_dir}/model.pt"
    local results_path="${output_dir}/results.json"
    local report_dir="${output_dir}/report"
    local holdout_path="${output_dir}/holdout_pairs.json"

    # Skip if results already exist
    if $SKIP_EXISTING && [[ -f "$results_path" ]]; then
        echo "  [SKIP] ${output_dir} (results.json exists)"
        return 0
    fi

    mkdir -p "$output_dir"

    log "Training: ${output_dir}"
    run_cmd uv run scripts/train_vae.py \
        --data "$data" \
        --tokenizer "$tokenizer" \
        --output "$model_path" \
        --seed "$seed" \
        --interp-holdout-pairs "$holdout_path" \
        "${BASELINE_ARGS[@]}" \
        "$@"

    log "Evaluating: ${output_dir}"
    if ! run_cmd uv run scripts/evaluate.py \
        --model "$model_path" \
        --tokenizer "$tokenizer" \
        --output "$report_dir" \
        --results-json "$results_path" \
        --interp-holdout-pairs "$holdout_path" \
        --seed "$seed" \
        --no-viz; then
        if $allow_eval_fail; then
            echo "  [WARN] Evaluation reported failing tests for ${output_dir}; continuing (allowed)."
        else
            return 1
        fi
    fi
}

# =============================================================================
# PREP: Download data and train tokenizers
# =============================================================================

should_run() {
    [[ "$SECTION" == "ALL" || "$SECTION" == "$1" ]]
}

if should_run "PREP" || [[ "$SECTION" == "ALL" ]]; then
    log "PREP: Downloading FineWeb data"

    for size in 2000000 4000000 8000000; do
        label=$(uv run python -c "
s=$size
if s>=1_000_000: print(f'{s//1_000_000}M')
elif s>=1_000: print(f'{s//1_000}K')
else: print(s)
")
        outfile="${DATA_DIR}/fineweb-${label}.txt"
        if $SKIP_EXISTING && [[ -f "$outfile" ]]; then
            echo "  [SKIP] ${outfile} (exists)"
        else
            run_cmd uv run scripts/prepare_data.py --num-chars "$size" --output "$outfile"
        fi
    done

    log "PREP: Training tokenizers"

    for vocab in 4000 8000 16000; do
        vlabel="${vocab}"
        if [[ $vocab -ge 1000 ]]; then
            vlabel="$((vocab / 1000))k"
        fi
        tok_prefix="${TOK_DIR}/fw-4M-v${vlabel}"
        if $SKIP_EXISTING && [[ -f "${tok_prefix}.model" ]]; then
            echo "  [SKIP] ${tok_prefix}.model (exists)"
        else
            run_cmd uv run scripts/train_tokenizer.py \
                --input "${DATA_DIR}/fineweb-4M.txt" \
                --output "$tok_prefix" \
                --vocab-size "$vocab"
        fi
    done
fi

# =============================================================================
# A. REPRODUCIBILITY — 5 seeds × baseline config (4M data, 16k vocab)
# =============================================================================

if should_run "A"; then
    log "SECTION A: Reproducibility (5 seeds)"

    DATA="${DATA_DIR}/fineweb-4M.txt"
    TOK="${TOK_DIR}/fw-4M-v16k.model"

    for seed in 42 123 456 789 1337; do
        train_and_eval \
            "${RESULTS_DIR}/A_reproducibility/seed_${seed}" \
            "$DATA" "$TOK" "$seed"
    done
fi

# =============================================================================
# B. DATA SCALE — 3 FineWeb sizes (2M, 4M, 8M), seed 42, 16k vocab
# =============================================================================

if should_run "B"; then
    log "SECTION B: Data Scale (2M, 4M, 8M)"

    TOK="${TOK_DIR}/fw-4M-v16k.model"

    for size in 2M 4M 8M; do
        train_and_eval \
            "${RESULTS_DIR}/B_data_scale/fineweb_${size}" \
            "${DATA_DIR}/fineweb-${size}.txt" "$TOK" 42
    done
fi

# =============================================================================
# C. VOCAB SCALE — 3 vocab sizes (4k, 8k, 16k), seed 42, 4M data
# =============================================================================

if should_run "C"; then
    log "SECTION C: Vocab Scale (4k, 8k, 16k)"

    DATA="${DATA_DIR}/fineweb-4M.txt"

    for vocab in 4k 8k 16k; do
        train_and_eval \
            "${RESULTS_DIR}/C_vocab_scale/vocab_${vocab}" \
            "$DATA" "${TOK_DIR}/fw-4M-v${vocab}.model" 42
    done
fi

# =============================================================================
# D. ABLATIONS — 8 configs via run_ablation.py, seed 42, 4M/16k
# =============================================================================

if should_run "D"; then
    log "SECTION D: Ablations (8 configs)"

    run_cmd uv run scripts/run_ablation.py \
        --data "${DATA_DIR}/fineweb-4M.txt" \
        --tokenizer "${TOK_DIR}/fw-4M-v16k.model" \
        --output-dir "${RESULTS_DIR}/D_ablations" \
        --seed 42
fi

# =============================================================================
# E+F. VERIFICATION — Check diffusion walk + geometry metrics exist
# =============================================================================

if should_run "EF"; then
    log "SECTION E+F: Verification (diffusion walk + geometry)"

    if $DRY_RUN; then
        echo "  [DRY RUN] uv run python: check Section A seed_42 results.json for Diffusion Walk + geometry"
    else
        uv run python - <<'PYEOF'
import json, sys

results_path = "validation_results/A_reproducibility/seed_42/results.json"
try:
    with open(results_path) as f:
        payload = json.load(f)
except FileNotFoundError:
    print(f"  [SKIP] {results_path} not found (run Section A first)")
    sys.exit(0)

if isinstance(payload, dict):
    results = payload.get("results", [])
else:
    results = payload

names = {r["name"] for r in results}

# Check diffusion walk
if "Diffusion Walk" in names:
    dw = next(r for r in results if r["name"] == "Diffusion Walk")
    status = "PASS" if dw["passed"] else "FAIL"
    print(f"  Diffusion Walk: {status} (score={dw['score']})")
else:
    print("  [WARN] Diffusion Walk test not found in results")

# Check geometry diagnostics (perturbation stability = local Lipschitz proxy)
if "Perturbation Stability" in names:
    ps = next(r for r in results if r["name"] == "Perturbation Stability")
    status = "PASS" if ps["passed"] else "FAIL"
    print(f"  Perturbation Stability (geometry): {status} (score={ps['score']})")
else:
    print("  [WARN] Perturbation Stability test not found in results")

# Check metric integrity
if "Metric Integrity" in names:
    mi = next(r for r in results if r["name"] == "Metric Integrity")
    status = "PASS" if mi["passed"] else "FAIL"
    print(f"  Metric Integrity: {status}")
else:
    print("  [WARN] Metric Integrity test not found in results")

print("  E+F verification complete.")
PYEOF
    fi
fi

# =============================================================================
# G. BASELINES — non-VAE and no-skipgram, seed 42, 4M/16k
# =============================================================================

if should_run "G"; then
    log "SECTION G: Baselines"

    DATA="${DATA_DIR}/fineweb-4M.txt"
    TOK="${TOK_DIR}/fw-4M-v16k.model"

    # non_vae: disable all VAE-specific losses
    train_and_eval \
        "${RESULTS_DIR}/G_baselines/non_vae" \
        "$DATA" "$TOK" 42 \
        --allow-eval-fail \
        --kl-weight 0 --kl-free-bits 0 \
        --prior-reg-weight 0 --interp-weight 0 --ifw-alpha 0

    # no_skipgram: full baseline but without skip-gram loss
    train_and_eval \
        "${RESULTS_DIR}/G_baselines/no_skipgram" \
        "$DATA" "$TOK" 42 \
        --allow-eval-fail \
        --skipgram-weight 0
fi

# =============================================================================
# H. AGGREGATION — Build summary.md from all results
# =============================================================================

if should_run "H" || [[ "$SECTION" == "ALL" ]]; then
    log "SECTION H: Aggregation"

    if $DRY_RUN; then
        echo "  [DRY RUN] uv run python: aggregate all results.json → validation_results/summary.md"
    else
        uv run python - <<'PYEOF'
import json, os, sys
from pathlib import Path

results_dir = Path("validation_results")
summary_path = results_dir / "summary.md"

def load_results(path):
    """Load results.json, return list of test dicts or None."""
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("results")
    return None

def fmt(val):
    """Format a value for the table."""
    if val is None:
        return "-"
    if isinstance(val, bool):
        return "PASS" if val else "FAIL"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

lines = ["# Hypothesis Validation Summary", ""]

# ---- Section A: Reproducibility ----
lines.append("## A. Reproducibility (5 seeds)")
lines.append("")
a_dir = results_dir / "A_reproducibility"
seeds = [42, 123, 456, 789, 1337]
a_results = {}
for seed in seeds:
    r = load_results(a_dir / f"seed_{seed}" / "results.json")
    if r:
        a_results[seed] = r

if a_results:
    # Get test names from first result
    test_names = [t["name"] for t in list(a_results.values())[0]]
    header = ["seed"] + test_names + ["all_pass"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for seed in seeds:
        if seed not in a_results:
            continue
        results = a_results[seed]
        row = [str(seed)]
        for tn in test_names:
            t = next((r for r in results if r["name"] == tn), None)
            if t:
                row.append(f"{fmt(t['score'])} {'P' if t['passed'] else 'F'}")
            else:
                row.append("-")
        all_pass = all(t["passed"] for t in results)
        row.append("YES" if all_pass else "NO")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
else:
    lines.append("No results found for Section A.")
    lines.append("")

# ---- Section B: Data Scale ----
lines.append("## B. Data Scale")
lines.append("")
b_dir = results_dir / "B_data_scale"
b_sizes = ["2M", "4M", "8M"]
b_results = {}
for size in b_sizes:
    r = load_results(b_dir / f"fineweb_{size}" / "results.json")
    if r:
        b_results[size] = r

if b_results:
    test_names = [t["name"] for t in list(b_results.values())[0]]
    header = ["data_size"] + test_names + ["all_pass"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for size in b_sizes:
        if size not in b_results:
            continue
        results = b_results[size]
        row = [size]
        for tn in test_names:
            t = next((r for r in results if r["name"] == tn), None)
            if t:
                row.append(f"{fmt(t['score'])} {'P' if t['passed'] else 'F'}")
            else:
                row.append("-")
        all_pass = all(t["passed"] for t in results)
        row.append("YES" if all_pass else "NO")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
else:
    lines.append("No results found for Section B.")
    lines.append("")

# ---- Section C: Vocab Scale ----
lines.append("## C. Vocab Scale")
lines.append("")
c_dir = results_dir / "C_vocab_scale"
c_vocabs = ["4k", "8k", "16k"]
c_results = {}
for vocab in c_vocabs:
    r = load_results(c_dir / f"vocab_{vocab}" / "results.json")
    if r:
        c_results[vocab] = r

if c_results:
    test_names = [t["name"] for t in list(c_results.values())[0]]
    header = ["vocab_size"] + test_names + ["all_pass"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for vocab in c_vocabs:
        if vocab not in c_results:
            continue
        results = c_results[vocab]
        row = [vocab]
        for tn in test_names:
            t = next((r for r in results if r["name"] == tn), None)
            if t:
                row.append(f"{fmt(t['score'])} {'P' if t['passed'] else 'F'}")
            else:
                row.append("-")
        all_pass = all(t["passed"] for t in results)
        row.append("YES" if all_pass else "NO")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
else:
    lines.append("No results found for Section C.")
    lines.append("")

# ---- Section D: Ablations ----
lines.append("## D. Ablations")
lines.append("")
d_dir = results_dir / "D_ablations"
d_configs = [
    "baseline", "no_marginal_entropy", "no_hhi", "no_marginal_no_hhi",
    "no_free_bits", "half_ifw", "no_ifw", "ifw_only",
]
d_results = {}
for cfg in d_configs:
    r = load_results(d_dir / cfg / "results.json")
    if r:
        d_results[cfg] = r

if d_results:
    test_names = [t["name"] for t in list(d_results.values())[0]]
    header = ["config"] + test_names + ["all_pass"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for cfg in d_configs:
        if cfg not in d_results:
            continue
        results = d_results[cfg]
        row = [cfg]
        for tn in test_names:
            t = next((r for r in results if r["name"] == tn), None)
            if t:
                row.append(f"{fmt(t['score'])} {'P' if t['passed'] else 'F'}")
            else:
                row.append("-")
        all_pass = all(t["passed"] for t in results)
        row.append("YES" if all_pass else "NO")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
else:
    lines.append("No results found for Section D.")
    lines.append("")

# ---- Section G: Baselines ----
lines.append("## G. Baselines")
lines.append("")
g_dir = results_dir / "G_baselines"
g_configs = ["non_vae", "no_skipgram"]
g_results = {}
for cfg in g_configs:
    r = load_results(g_dir / cfg / "results.json")
    if r:
        g_results[cfg] = r

if g_results:
    test_names = [t["name"] for t in list(g_results.values())[0]]
    header = ["config"] + test_names + ["all_pass"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for cfg in g_configs:
        if cfg not in g_results:
            continue
        results = g_results[cfg]
        row = [cfg]
        for tn in test_names:
            t = next((r for r in results if r["name"] == tn), None)
            if t:
                row.append(f"{fmt(t['score'])} {'P' if t['passed'] else 'F'}")
            else:
                row.append("-")
        all_pass = all(t["passed"] for t in results)
        row.append("YES" if all_pass else "NO")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
else:
    lines.append("No results found for Section G.")
    lines.append("")

# ---- Summary ----
lines.append("## Overall Summary")
lines.append("")
total_runs = 0
total_pass = 0
for section_results in [a_results, b_results, c_results, d_results, g_results]:
    for key, results in section_results.items():
        total_runs += 1
        if all(t["passed"] for t in results):
            total_pass += 1

lines.append(f"- **Total runs with results**: {total_runs}")
lines.append(f"- **All tests passing**: {total_pass}/{total_runs}")
lines.append("")

summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w") as f:
    f.write("\n".join(lines))

print(f"  Validation summary written to: {summary_path}")
print(f"  Runs found: {total_runs}, all-pass: {total_pass}/{total_runs}")
PYEOF
    fi
fi

log "Done!"
if $DRY_RUN; then
    echo "  (dry run — no commands were executed)"
fi
