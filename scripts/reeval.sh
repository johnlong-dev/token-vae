#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Token VAE — Re-evaluation Script
#
# Re-runs the full evaluation suite on every trained model found under
# validation_results/, regenerating results.json, reports, and summary.md.
# Skips data download and model training entirely.
#
# Usage:
#   ./scripts/reeval.sh                     # re-evaluate all runs
#   ./scripts/reeval.sh --dry-run           # print commands without executing
#   ./scripts/reeval.sh --section A         # re-evaluate only section A
#   ./scripts/reeval.sh --parallel 4        # run up to 4 evals concurrently
# =============================================================================

RESULTS_DIR="validation_results"
TOK_DIR="artifacts/tokenizer"

# --- CLI parsing ---
DRY_RUN=false
SECTION="ALL"
PARALLEL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --section)    SECTION="$2"; shift 2 ;;
        --parallel)   PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helpers ---

log() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
}

should_run() {
    [[ "$SECTION" == "ALL" || "$SECTION" == "$1" ]]
}

TOTAL=0
PASS=0
FAIL=0
SKIP=0

# Background job tracking for --parallel
JOBS=()
JOB_DIRS=()

wait_for_slot() {
    while (( ${#JOBS[@]} >= PARALLEL )); do
        local new_jobs=()
        local new_dirs=()
        for i in "${!JOBS[@]}"; do
            if kill -0 "${JOBS[$i]}" 2>/dev/null; then
                new_jobs+=("${JOBS[$i]}")
                new_dirs+=("${JOB_DIRS[$i]}")
            else
                wait "${JOBS[$i]}" || true
            fi
        done
        JOBS=("${new_jobs[@]+"${new_jobs[@]}"}")
        JOB_DIRS=("${new_dirs[@]+"${new_dirs[@]}"}")
        if (( ${#JOBS[@]} >= PARALLEL )); then
            sleep 1
        fi
    done
}

wait_all() {
    for pid in "${JOBS[@]+"${JOBS[@]}"}"; do
        wait "$pid" || true
    done
    JOBS=()
    JOB_DIRS=()
}

eval_run() {
    # Usage: eval_run <run_dir> <tokenizer> <seed> [--allow-eval-fail]
    local run_dir="$1"; shift
    local tokenizer="$1"; shift
    local seed="$1"; shift
    local allow_eval_fail=false
    if [[ $# -gt 0 && "$1" == "--allow-eval-fail" ]]; then
        allow_eval_fail=true
        shift
    fi

    local model_path="${run_dir}/model.pt"
    local results_path="${run_dir}/results.json"
    local report_dir="${run_dir}/report"
    local holdout_path="${run_dir}/holdout_pairs.json"

    if [[ ! -f "$model_path" ]]; then
        echo "  [SKIP] ${run_dir} (no model.pt)"
        (( SKIP++ )) || true
        return 0
    fi

    if [[ ! -f "$tokenizer" ]]; then
        echo "  [SKIP] ${run_dir} (tokenizer not found: ${tokenizer})"
        (( SKIP++ )) || true
        return 0
    fi

    TOTAL=$((TOTAL + 1))

    local holdout_args=()
    if [[ -f "$holdout_path" ]]; then
        holdout_args=(--interp-holdout-pairs "$holdout_path")
    fi

    echo "  Evaluating: ${run_dir}"

    if $DRY_RUN; then
        echo "  [DRY RUN] uv run scripts/evaluate.py --model $model_path --tokenizer $tokenizer --output $report_dir --results-json $results_path --seed $seed ${holdout_args[*]+"${holdout_args[*]}"} --no-viz"
        return 0
    fi

    if uv run scripts/evaluate.py \
        --model "$model_path" \
        --tokenizer "$tokenizer" \
        --output "$report_dir" \
        --results-json "$results_path" \
        --seed "$seed" \
        "${holdout_args[@]+"${holdout_args[@]}"}" \
        --no-viz; then
        (( PASS++ )) || true
    else
        if $allow_eval_fail; then
            echo "  [WARN] Evaluation reported failing tests for ${run_dir}; continuing."
            (( FAIL++ )) || true
        else
            (( FAIL++ )) || true
            return 1
        fi
    fi
}

eval_run_bg() {
    # Launch eval_run in background, respecting --parallel
    if (( PARALLEL <= 1 )); then
        eval_run "$@"
        return
    fi
    wait_for_slot
    eval_run "$@" &
    JOBS+=($!)
    JOB_DIRS+=("$1")
}

# =============================================================================
# A. REPRODUCIBILITY — 5 seeds
# =============================================================================

if should_run "A"; then
    log "SECTION A: Reproducibility (5 seeds)"
    TOK="${TOK_DIR}/fw-4M-v16k.model"
    for seed in 42 123 456 789 1337; do
        eval_run_bg "${RESULTS_DIR}/A_reproducibility/seed_${seed}" "$TOK" "$seed"
    done
    wait_all
fi

# =============================================================================
# B. DATA SCALE — 2M, 4M, 8M
# =============================================================================

if should_run "B"; then
    log "SECTION B: Data Scale (2M, 4M, 8M)"
    TOK="${TOK_DIR}/fw-4M-v16k.model"
    for size in 2M 4M 8M; do
        eval_run_bg "${RESULTS_DIR}/B_data_scale/fineweb_${size}" "$TOK" 42
    done
    wait_all
fi

# =============================================================================
# C. VOCAB SCALE — 4k, 8k, 16k
# =============================================================================

if should_run "C"; then
    log "SECTION C: Vocab Scale (4k, 8k, 16k)"
    for vocab in 4k 8k 16k; do
        eval_run_bg "${RESULTS_DIR}/C_vocab_scale/vocab_${vocab}" \
            "${TOK_DIR}/fw-4M-v${vocab}.model" 42
    done
    wait_all
fi

# =============================================================================
# D. ABLATIONS — 8 configs
# =============================================================================

if should_run "D"; then
    log "SECTION D: Ablations (8 configs)"
    TOK="${TOK_DIR}/fw-4M-v16k.model"
    for cfg in baseline no_marginal_entropy no_hhi no_marginal_no_hhi \
               no_free_bits half_ifw no_ifw ifw_only; do
        eval_run_bg "${RESULTS_DIR}/D_ablations/${cfg}" "$TOK" 42
    done
    wait_all
fi

# =============================================================================
# G. BASELINES — non_vae, no_skipgram
# =============================================================================

if should_run "G"; then
    log "SECTION G: Baselines"
    TOK="${TOK_DIR}/fw-4M-v16k.model"
    for cfg in non_vae no_skipgram; do
        eval_run_bg "${RESULTS_DIR}/G_baselines/${cfg}" "$TOK" 42 --allow-eval-fail
    done
    wait_all
fi

# =============================================================================
# H. AGGREGATION — Rebuild summary.md
# =============================================================================

if should_run "H" || [[ "$SECTION" == "ALL" ]]; then
    log "SECTION H: Aggregation"

    if $DRY_RUN; then
        echo "  [DRY RUN] aggregate all results.json -> validation_results/summary.md"
    else
        uv run python - <<'PYEOF'
import json
from pathlib import Path

results_dir = Path("validation_results")
summary_path = results_dir / "summary.md"


def load_results(path):
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

# =============================================================================
# Done
# =============================================================================

log "Re-evaluation complete"
if $DRY_RUN; then
    echo "  (dry run -- no commands were executed)"
else
    echo "  Evaluated: ${TOTAL}  Passed: ${PASS}  Failed: ${FAIL}  Skipped: ${SKIP}"
fi
