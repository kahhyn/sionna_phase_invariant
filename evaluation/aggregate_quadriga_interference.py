"""Aggregate QuaDRiGa interference results across trajectories and seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_NAME = "quadriga_interference_summary.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["train_profile"],
        row["receiver_label"],
        row["model"],
        row["snr_db"],
        row["sir_db"],
        row["interference_mode"],
        row["sir_normalization"],
        row["n0_mode"],
        row["phase_mode"],
        row["channel_normalization"],
    )


def pair_key(row: dict[str, str], source: Path) -> tuple[str, ...]:
    return (
        str(source),
        row["train_profile"],
        row["train_seed"],
        row["eval_seed"],
        row["snr_db"],
        row["sir_db"],
        row["interference_mode"],
        row["sir_normalization"],
        row["n0_mode"],
        row["phase_mode"],
        row["channel_normalization"],
    )


def mean_std(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) > 1:
        std = statistics.stdev(values)
        sem = std / math.sqrt(len(values))
    else:
        std = 0.0
        sem = 0.0
    return mean, std, sem


def aggregate(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(input_root.rglob(SUMMARY_NAME))
    if not files:
        raise FileNotFoundError(f"No {SUMMARY_NAME} files below {input_root}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    pairs: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                groups[condition_key(row)].append(row)
                pairs[pair_key(row, path)][row["receiver_label"]] = row

    aggregate_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        (
            train_profile,
            receiver_label,
            model,
            snr_db,
            sir_db,
            interference_mode,
            sir_normalization,
            n0_mode,
            phase_mode,
            channel_normalization,
        ) = key
        errors = sum(int(row["bit_errors"]) for row in rows)
        bits = sum(int(row["valid_bits"]) for row in rows)
        bce_weighted = sum(
            float(row["bce"]) * int(row["valid_bits"]) for row in rows
        )
        ber_values = [float(row["ber"]) for row in rows]
        bce_values = [float(row["bce"]) for row in rows]
        ber_mean, ber_std, ber_sem = mean_std(ber_values)
        bce_mean, bce_std, bce_sem = mean_std(bce_values)
        aggregate_rows.append(
            {
                "train_profile": train_profile,
                "receiver_label": receiver_label,
                "model": model,
                "snr_db": snr_db,
                "sir_db": sir_db,
                "interference_mode": interference_mode,
                "sir_normalization": sir_normalization,
                "n0_mode": n0_mode,
                "phase_mode": phase_mode,
                "channel_normalization": channel_normalization,
                "num_runs": len(rows),
                "num_train_seeds": len({row["train_seed"] for row in rows}),
                "ber_mean": ber_mean,
                "ber_std": ber_std,
                "ber_sem": ber_sem,
                "ber_pooled": errors / bits,
                "bce_mean": bce_mean,
                "bce_std": bce_std,
                "bce_sem": bce_sem,
                "bce_pooled": bce_weighted / bits,
                "bit_errors": errors,
                "valid_bits": bits,
                "h_hat_nmse_db_mean": statistics.fmean(
                    float(row["h_hat_nmse_db"]) for row in rows
                ),
                "achieved_sir_db_mean": statistics.fmean(
                    float(row["achieved_sir_db"]) for row in rows
                ),
                "achieved_sinr_db_mean": statistics.fmean(
                    float(row["achieved_sinr_db"]) for row in rows
                ),
            }
        )

    paired_groups: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    for key, receiver_rows in pairs.items():
        if set(receiver_rows) != {"A_phase_invariant", "C_strict_matched"}:
            raise ValueError(f"Incomplete A/C pair for {key}: {sorted(receiver_rows)}")
        a_row = receiver_rows["A_phase_invariant"]
        c_row = receiver_rows["C_strict_matched"]
        aggregate_key = (
            a_row["train_profile"],
            a_row["snr_db"],
            a_row["sir_db"],
            a_row["interference_mode"],
            a_row["sir_normalization"],
            a_row["n0_mode"],
            a_row["phase_mode"],
            a_row["channel_normalization"],
        )
        paired_groups[aggregate_key].append(
            {
                "ber_delta": float(a_row["ber"]) - float(c_row["ber"]),
                "bce_delta": float(a_row["bce"]) - float(c_row["bce"]),
                "a_ber": float(a_row["ber"]),
                "c_ber": float(c_row["ber"]),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for key, rows in sorted(paired_groups.items()):
        (
            train_profile,
            snr_db,
            sir_db,
            interference_mode,
            sir_normalization,
            n0_mode,
            phase_mode,
            channel_normalization,
        ) = key
        ber_delta_mean, ber_delta_std, ber_delta_sem = mean_std(
            [row["ber_delta"] for row in rows]
        )
        bce_delta_mean, bce_delta_std, bce_delta_sem = mean_std(
            [row["bce_delta"] for row in rows]
        )
        paired_rows.append(
            {
                "train_profile": train_profile,
                "snr_db": snr_db,
                "sir_db": sir_db,
                "interference_mode": interference_mode,
                "sir_normalization": sir_normalization,
                "n0_mode": n0_mode,
                "phase_mode": phase_mode,
                "channel_normalization": channel_normalization,
                "num_pairs": len(rows),
                "a_ber_mean": statistics.fmean(row["a_ber"] for row in rows),
                "c_ber_mean": statistics.fmean(row["c_ber"] for row in rows),
                "a_minus_c_ber_mean": ber_delta_mean,
                "a_minus_c_ber_std": ber_delta_std,
                "a_minus_c_ber_sem": ber_delta_sem,
                "a_minus_c_bce_mean": bce_delta_mean,
                "a_minus_c_bce_std": bce_delta_std,
                "a_minus_c_bce_sem": bce_delta_sem,
            }
        )

    return aggregate_rows, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--paired_out_csv", required=True)
    args = parser.parse_args()
    aggregate_rows, paired_rows = aggregate(Path(args.input_root))
    write_csv(Path(args.out_csv), aggregate_rows)
    write_csv(Path(args.paired_out_csv), paired_rows)
    print(f"Aggregated {len(aggregate_rows)} receiver conditions to {args.out_csv}")
    print(f"Aggregated {len(paired_rows)} paired conditions to {args.paired_out_csv}")


if __name__ == "__main__":
    main()
