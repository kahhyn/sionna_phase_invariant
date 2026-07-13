"""Aggregate oracle few-shot adaptation results across training seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "target_post_ber",
    "source_post_ber",
    "target_ber_improvement",
    "source_ber_forgetting",
)


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    files = sorted(run_root.glob("**/fewshot_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No fewshot_summary.csv files under {run_root}")

    grouped = defaultdict(list)
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["model"],
                    row["target_tdl_model"],
                    float(row["target_delay_ns"]),
                    int(row["budget"]),
                )
                grouped[key].append(row)

    output_rows = []
    for key, rows in sorted(grouped.items()):
        model, target_tdl_model, target_delay_ns, budget = key
        result = {
            "model": model,
            "target_tdl_model": target_tdl_model,
            "target_delay_ns": target_delay_ns,
            "budget": budget,
            "num_train_seeds": len(rows),
            "parameter_count": int(rows[0]["parameter_count"]),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(values)
        output_rows.append(result)

    by_domain = defaultdict(list)
    for row in output_rows:
        key = (row["model"], row["target_tdl_model"], row["target_delay_ns"])
        by_domain[key].append(row)

    for rows in by_domain.values():
        rows.sort(key=lambda row: row["budget"])
        pre_ber = next(
            row["target_post_ber_mean"] for row in rows if row["budget"] == 0
        )
        best_ber = min(row["target_post_ber_mean"] for row in rows)
        gap = pre_ber - best_ber
        threshold = pre_ber - 0.9 * gap
        n90 = 0 if gap <= 0 else next(
            (
                row["budget"]
                for row in rows
                if row["target_post_ber_mean"] <= threshold
            ),
            -1,
        )
        for row in rows:
            row["target_pre_ber_mean"] = pre_ber
            row["target_best_ber_mean"] = best_ber
            row["recovery_fraction_to_best_mean_curve"] = (
                (pre_ber - row["target_post_ber_mean"]) / gap if gap > 0 else 0.0
            )
            row["n90_samples_mean_curve"] = n90

    out_path = (
        Path(args.out_csv)
        if args.out_csv
        else run_root / "fewshot_multiseed_summary.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Aggregated {len(files)} files into {out_path}")


if __name__ == "__main__":
    main()
