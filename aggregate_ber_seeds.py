"""Aggregate BER CSV files across training and evaluation seeds."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for filename in args.input_files:
        path = Path(filename)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                required = {"model", "train_seed", "eval_seed", "snr_db"}
                if not required.issubset(row):
                    missing = ", ".join(sorted(required.difference(row)))
                    raise ValueError(f"{path} lacks metadata columns: {missing}")
                parsed = {
                    "ber": float(row["ber"]),
                    "bce": float(row["bce"]),
                    "bit_errors": int(row["bit_errors"]),
                    "valid_bits": int(row["valid_bits"]),
                    "train_seed": int(row["train_seed"]),
                    "eval_seed": int(row["eval_seed"]),
                }
                key = (row["model"], float(row["snr_db"]))
                grouped[key].append(parsed)

    output_rows = []
    for (model, snr_db), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        bers = [value["ber"] for value in values]
        bces = [value["bce"] for value in values]
        num_runs = len(values)
        ber_std = stdev(bers) if num_runs > 1 else 0.0
        ber_sem = ber_std / math.sqrt(num_runs) if num_runs > 1 else 0.0
        total_errors = sum(value["bit_errors"] for value in values)
        total_bits = sum(value["valid_bits"] for value in values)
        output_rows.append(
            {
                "model": model,
                "snr_db": snr_db,
                "num_runs": num_runs,
                "num_train_seeds": len({value["train_seed"] for value in values}),
                "num_eval_seeds": len({value["eval_seed"] for value in values}),
                "ber_mean": fmean(bers),
                "ber_std": ber_std,
                "ber_sem": ber_sem,
                "ber_pooled": total_errors / total_bits,
                "bce_mean": fmean(bces),
                "total_bit_errors": total_errors,
                "total_valid_bits": total_bits,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "snr_db",
        "num_runs",
        "num_train_seeds",
        "num_eval_seeds",
        "ber_mean",
        "ber_std",
        "ber_sem",
        "ber_pooled",
        "bce_mean",
        "total_bit_errors",
        "total_valid_bits",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Aggregated {len(args.input_files)} files into {out_path}")


if __name__ == "__main__":
    main()
