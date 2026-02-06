#!/usr/bin/env python3
"""Ablation study harness for Token VAE.

Runs training + evaluation for multiple configurations and produces a
comparison table. Each config is a set of CLI overrides on top of the
baseline (matching run_validation.sh defaults).

Usage:
    uv run scripts/run_ablation.py --seed 42
    uv run scripts/run_ablation.py --configs baseline no_ifw --seed 42
    uv run scripts/run_ablation.py --dry-run
    uv run scripts/run_ablation.py --seeds 42 123 456
    uv run scripts/run_ablation.py --aggregate-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Base training args matching run_validation.sh
BASE_TRAIN_ARGS = [
    "--interp-weight", "0.05",
    "--interp-pairs", "64",
    "--interp-entropy-target", "8.0",
    "--interp-holdout-num", "200",
    "--identity-vocab-repeats", "5",
    "--kl-weight", "0.03",
    "--kl-free-bits", "0.25",
    "--ifw-alpha", "1.0",
    "--prior-reg-weight", "0.50",
    "--prior-reg-interval", "1",
    "--prior-entropy-target", "4.0",
    "--prior-max-freq-target", "0.05",
    "--prior-marginal-entropy-target", "7.0",
    "--prior-hhi-target", "0.001",
]

# Ablation configs: name -> list of CLI overrides
ABLATION_CONFIGS = {
    "baseline": [],
    "no_marginal_entropy": [
        "--prior-marginal-entropy-target", "0.0",
    ],
    "no_hhi": [
        "--prior-hhi-target", "0.0",
    ],
    "no_marginal_no_hhi": [
        "--prior-marginal-entropy-target", "0.0",
        "--prior-hhi-target", "0.0",
    ],
    "no_free_bits": [
        "--kl-free-bits", "0.0",
    ],
    "half_ifw": [
        "--ifw-alpha", "0.5",
    ],
    "no_ifw": [
        "--ifw-alpha", "0.0",
    ],
    "ifw_only": [
        "--ifw-alpha", "1.0",
        "--kl-free-bits", "0.0",
        "--prior-reg-weight", "0.0",
        "--prior-marginal-entropy-target", "0.0",
        "--prior-hhi-target", "0.0",
    ],
}


def build_train_cmd(
    config_name: str, seed: int, output_dir: Path,
    data: str | None = None, tokenizer: str | None = None,
) -> list[str]:
    """Build the training command for a given config."""
    model_path = output_dir / "model.pt"
    holdout_path = output_dir / "holdout_pairs.json"
    overrides = ABLATION_CONFIGS[config_name]

    # Start with base, then apply overrides (later values win)
    args = list(BASE_TRAIN_ARGS)
    # Parse overrides as key-value pairs and replace in base
    override_keys = set()
    for i in range(0, len(overrides), 2):
        override_keys.add(overrides[i])
    # Remove overridden keys from base
    filtered = []
    i = 0
    while i < len(args):
        if args[i] in override_keys:
            i += 2  # skip key and value
        else:
            filtered.append(args[i])
            i += 1
    filtered.extend(overrides)

    cmd = [
        "uv", "run", "scripts/train_vae.py",
        "--output", str(model_path),
        "--seed", str(seed),
        "--interp-holdout-pairs", str(holdout_path),
        "--interp-holdout-seed", str(seed),
    ]
    if data:
        cmd.extend(["--data", data])
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    cmd.extend(filtered)

    return cmd


def build_eval_cmd(output_dir: Path, seed: int, tokenizer: str | None = None) -> list[str]:
    """Build the evaluation command."""
    model_path = output_dir / "model.pt"
    results_path = output_dir / "results.json"
    report_dir = output_dir / "report"
    holdout_path = output_dir / "holdout_pairs.json"

    cmd = [
        "uv", "run", "scripts/evaluate.py",
        "--model", str(model_path),
        "--output", str(report_dir),
        "--results-json", str(results_path),
        "--interp-holdout-pairs", str(holdout_path),
        "--seed", str(seed),
        "--no-viz",
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    return cmd


def run_config(
    config_name: str, seed: int, base_dir: Path, dry_run: bool = False,
    data: str | None = None, tokenizer: str | None = None,
) -> bool:
    """Run training + evaluation for a single config.

    Returns True if both commands succeed.
    """
    if seed != 42:
        output_dir = base_dir / f"{config_name}_seed{seed}"
    else:
        output_dir = base_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = build_train_cmd(config_name, seed, output_dir, data=data, tokenizer=tokenizer)
    eval_cmd = build_eval_cmd(output_dir, seed=seed, tokenizer=tokenizer)

    if dry_run:
        print(f"\n[{config_name}] (seed={seed})")
        print(f"  TRAIN: {' '.join(train_cmd)}")
        print(f"  EVAL:  {' '.join(eval_cmd)}")
        return True

    print(f"\n{'='*60}")
    print(f"Running: {config_name} (seed={seed})")
    print(f"{'='*60}")

    # Train
    print(f"\n--- Training {config_name} ---")
    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print(f"FAILED: Training for {config_name} (exit code {result.returncode})")
        return False

    # Evaluate
    print(f"\n--- Evaluating {config_name} ---")
    result = subprocess.run(eval_cmd)
    if result.returncode != 0:
        print(f"NOTE: Evaluation for {config_name} reported failures (exit code {result.returncode})")
        # Don't return False — eval "failure" means tests didn't pass, not a crash

    return True


def load_results(results_path: Path) -> list[dict] | None:
    """Load results JSON from a config directory."""
    if not results_path.exists():
        return None
    with open(results_path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return None


def aggregate_results(base_dir: Path, configs: list[str], seeds: list[int]) -> str:
    """Produce a markdown comparison table from results."""
    lines = [
        "# Ablation Study Results",
        "",
        f"Base directory: `{base_dir}`",
        "",
    ]

    # Collect all results
    all_data: dict[str, list[dict]] = {}

    for config in configs:
        config_results = []
        for seed in seeds:
            if len(seeds) == 1 and seed == 42:
                result_path = base_dir / config / "results.json"
            else:
                result_path = base_dir / f"{config}_seed{seed}" / "results.json"
            results = load_results(result_path)
            if results:
                config_results.append(results)
        if config_results:
            all_data[config] = config_results

    if not all_data:
        lines.append("No results found.")
        return "\n".join(lines)

    # Determine all test names from any config
    test_names = []
    for results_list in all_data.values():
        for test in results_list[0]:
            if test["name"] not in test_names:
                test_names.append(test["name"])

    # Key metrics to extract per test
    metric_keys = {
        "Prior Decodability": [
            ("score", "non_special"),
            ("details.unique_fraction", "unique_frac"),
            ("details.entropy_mean", "prior_H"),
        ],
        "Perturbation Stability": [
            ("score", "overlap_σ=0.05"),
        ],
        "Interpolation Continuity": [
            ("score", "mean_js"),
            ("details.max_entropy", "max_H"),
        ],
        "Reconstruction Accuracy": [
            ("score", "top1_acc"),
            ("details.collapsed_fraction", "collapsed"),
        ],
        "Metric Integrity": [
            ("passed", "integrity"),
        ],
        "Diffusion Walk": [
            ("score", "min_non_special"),
            ("details.max_entropy_mean", "max_H"),
        ],
    }

    # Build a flat table: config -> {metric_label: value_str}
    def extract_value(result: dict, key: str):
        """Extract a nested value like 'details.unique_fraction'."""
        parts = key.split(".")
        obj = result
        for p in parts:
            if isinstance(obj, dict):
                obj = obj.get(p)
            else:
                return None
        return obj

    # Collect all column labels
    columns = ["config", "seeds", "all_pass"]
    for test_name in test_names:
        for _, label in metric_keys.get(test_name, [("score", test_name)]):
            columns.append(label)

    table_rows = []
    for config, results_list in all_data.items():
        row = {"config": config, "seeds": str(len(results_list))}

        # For each metric, collect values across seeds
        all_passed_list = []
        for test_name in test_names:
            keys = metric_keys.get(test_name, [("score", test_name)])
            for key_path, label in keys:
                values = []
                for results in results_list:
                    for test in results:
                        if test["name"] == test_name:
                            val = extract_value(test, key_path)
                            if val is not None:
                                values.append(val)
                            break

                if not values:
                    row[label] = "-"
                elif all(isinstance(v, bool) for v in values):
                    pass_count = sum(1 for v in values if v)
                    row[label] = f"{pass_count}/{len(values)}"
                elif len(values) == 1:
                    row[label] = f"{values[0]:.4f}" if isinstance(values[0], float) else str(values[0])
                else:
                    import numpy as np
                    mean = np.mean(values)
                    std = np.std(values)
                    row[label] = f"{mean:.4f}±{std:.4f}"

            # Track all_pass
            for results in results_list:
                for test in results:
                    if test["name"] == test_name:
                        all_passed_list.append(test["passed"])

        row["all_pass"] = "YES" if all(all_passed_list) else "NO"
        table_rows.append(row)

    # Render markdown table
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in table_rows:
        cells = [str(row.get(col, "-")) for col in columns]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Token VAE Ablation Study")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Specific configs to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all runs (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Run each config with multiple seeds (e.g. --seeds 42 123 456)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training/eval, just aggregate existing results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/ablations",
        help="Base output directory",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data (passed to train_vae.py)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer model (passed to train_vae.py and evaluate.py)",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    configs = args.configs or list(ABLATION_CONFIGS.keys())
    seeds = args.seeds or [args.seed]

    # Validate config names
    for c in configs:
        if c not in ABLATION_CONFIGS:
            print(f"Unknown config: {c}")
            print(f"Available: {', '.join(ABLATION_CONFIGS.keys())}")
            return 1

    if not args.aggregate_only:
        # Run each config
        for config in configs:
            for seed in seeds:
                success = run_config(
                    config, seed, base_dir, dry_run=args.dry_run,
                    data=args.data, tokenizer=args.tokenizer,
                )
                if not success and not args.dry_run:
                    print(f"\nAborting due to training failure in {config}")
                    return 1

    if args.dry_run:
        return 0

    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}")

    comparison = aggregate_results(base_dir, configs, seeds)
    comparison_path = base_dir / "comparison.md"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        f.write(comparison)

    print(f"\nComparison table saved to: {comparison_path}")
    print()
    print(comparison)

    return 0


if __name__ == "__main__":
    sys.exit(main())
