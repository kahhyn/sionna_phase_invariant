"""Aggregate DICHASUS receiver results across antennas and training seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_NAME = "dichasus_summary.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    return mean, std, std / math.sqrt(len(values))


def condition_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["dataset_id"],
        row["train_profile"],
        row["receiver_label"],
        row["model"],
        row["snr_db"],
        row["phase_mode"],
        row["channel_normalization"],
        row["position_source"],
        row["start_record"],
        row["record_stride"],
        row["max_records"],
        row["min_measured_snr_db_filter"],
    )


def pair_key(row: dict[str, str], path: Path) -> tuple[str, ...]:
    return (
        str(path),
        row["dataset_id"],
        row["train_profile"],
        row["train_seed"],
        row["antenna_index"],
        row["eval_seed"],
        row["snr_db"],
        row["phase_mode"],
        row["channel_normalization"],
    )


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
            dataset_id,
            train_profile,
            receiver_label,
            model,
            snr_db,
            phase_mode,
            normalization,
            position_source,
            start_record,
            record_stride,
            max_records,
            min_measured_snr_db_filter,
        ) = key
        errors = sum(int(row["bit_errors"]) for row in rows)
        bits = sum(int(row["valid_bits"]) for row in rows)
        bce_sum = sum(float(row["bce"]) * int(row["valid_bits"]) for row in rows)
        ber_mean, ber_std, ber_sem = mean_std([float(row["ber"]) for row in rows])
        bce_mean, bce_std, bce_sem = mean_std([float(row["bce"]) for row in rows])
        aggregate_rows.append(
            {
                "dataset_id": dataset_id,
                "train_profile": train_profile,
                "receiver_label": receiver_label,
                "model": model,
                "snr_db": snr_db,
                "phase_mode": phase_mode,
                "channel_normalization": normalization,
                "position_source": position_source,
                "start_record": start_record,
                "record_stride": record_stride,
                "max_records": max_records,
                "min_measured_snr_db_filter": min_measured_snr_db_filter,
                "num_runs": len(rows),
                "num_train_seeds": len({row["train_seed"] for row in rows}),
                "num_antennas": len({row["antenna_index"] for row in rows}),
                "ber_mean": ber_mean,
                "ber_std": ber_std,
                "ber_sem": ber_sem,
                "ber_pooled": errors / bits,
                "bce_mean": bce_mean,
                "bce_std": bce_std,
                "bce_sem": bce_sem,
                "bce_pooled": bce_sum / bits,
                "bit_errors": errors,
                "valid_bits": bits,
                "h_hat_nmse_db_mean": statistics.fmean(
                    float(row["h_hat_nmse_db"]) for row in rows
                ),
                "measured_snr_db_mean": statistics.fmean(
                    float(row["measured_snr_db_mean"]) for row in rows
                ),
            }
        )

    paired: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    for key, receiver_rows in pairs.items():
        if set(receiver_rows) != {"A_phase_invariant", "C_strict_matched"}:
            raise ValueError(f"Incomplete A/C pair for {key}: {sorted(receiver_rows)}")
        a = receiver_rows["A_phase_invariant"]
        c = receiver_rows["C_strict_matched"]
        group_key = (
            a["dataset_id"],
            a["train_profile"],
            a["snr_db"],
            a["phase_mode"],
            a["channel_normalization"],
            a["position_source"],
            a["start_record"],
            a["record_stride"],
            a["max_records"],
            a["min_measured_snr_db_filter"],
        )
        paired[group_key].append(
            {
                "a_ber": float(a["ber"]),
                "c_ber": float(c["ber"]),
                "ber_delta": float(a["ber"]) - float(c["ber"]),
                "bce_delta": float(a["bce"]) - float(c["bce"]),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for key, rows in sorted(paired.items()):
        (
            dataset_id,
            train_profile,
            snr_db,
            phase_mode,
            normalization,
            position_source,
            start_record,
            record_stride,
            max_records,
            min_measured_snr_db_filter,
        ) = key
        ber_delta_mean, ber_delta_std, ber_delta_sem = mean_std(
            [row["ber_delta"] for row in rows]
        )
        bce_delta_mean, bce_delta_std, bce_delta_sem = mean_std(
            [row["bce_delta"] for row in rows]
        )
        paired_rows.append(
            {
                "dataset_id": dataset_id,
                "train_profile": train_profile,
                "snr_db": snr_db,
                "phase_mode": phase_mode,
                "channel_normalization": normalization,
                "position_source": position_source,
                "start_record": start_record,
                "record_stride": record_stride,
                "max_records": max_records,
                "min_measured_snr_db_filter": min_measured_snr_db_filter,
                "num_pairs": len(rows),
                "a_wins": sum(row["ber_delta"] < 0 for row in rows),
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
    print(f"Aggregated {len(aggregate_rows)} conditions to {args.out_csv}")
    print(f"Aggregated {len(paired_rows)} paired conditions to {args.paired_out_csv}")


if __name__ == "__main__":
    main()
