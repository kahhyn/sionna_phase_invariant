"""Aggregate 5G LDPC BLER CSV files across training/evaluation seeds."""

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
        with Path(filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                value = {
                    "bler": float(row["bler"]),
                    "post_ber": float(row["post_ldpc_ber"]),
                    "pre_ber": float(row["pre_ldpc_coded_ber"]),
                    "bce": float(row["coded_bce"]),
                    "block_errors": int(row["block_errors"]),
                    "num_blocks": int(row["num_blocks"]),
                    "info_bit_errors": int(row["info_bit_errors"]),
                    "coded_bit_errors": int(row["coded_bit_errors"]),
                    "k": int(row["k"]),
                    "n": int(row["n"]),
                    "train_seed": int(row["train_seed"]),
                    "eval_seed": int(row["eval_seed"]),
                    "target_reached": int(row["target_reached"]),
                }
                grouped[(row["model"], float(row["ebno_db"]))].append(value)

    output = []
    for (model, ebno_db), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        blers = [value["bler"] for value in values]
        num_runs = len(values)
        bler_std = stdev(blers) if num_runs > 1 else 0.0
        total_blocks = sum(value["num_blocks"] for value in values)
        total_block_errors = sum(value["block_errors"] for value in values)
        total_info_bits = sum(value["num_blocks"] * value["k"] for value in values)
        total_coded_bits = sum(value["num_blocks"] * value["n"] for value in values)
        output.append(
            {
                "model": model,
                "ebno_db": ebno_db,
                "num_runs": num_runs,
                "num_train_seeds": len({v["train_seed"] for v in values}),
                "num_eval_seeds": len({v["eval_seed"] for v in values}),
                "bler_mean": fmean(blers),
                "bler_std": bler_std,
                "bler_sem": bler_std / math.sqrt(num_runs) if num_runs > 1 else 0.0,
                "bler_pooled": total_block_errors / total_blocks,
                "post_ldpc_ber_pooled": sum(v["info_bit_errors"] for v in values) / total_info_bits,
                "pre_ldpc_coded_ber_pooled": sum(v["coded_bit_errors"] for v in values) / total_coded_bits,
                "coded_bce_mean": fmean(v["bce"] for v in values),
                "total_block_errors": total_block_errors,
                "total_blocks": total_blocks,
                "target_reached_runs": sum(v["target_reached"] for v in values),
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    print(f"Aggregated {len(args.input_files)} files into {out_path}")


if __name__ == "__main__":
    main()
