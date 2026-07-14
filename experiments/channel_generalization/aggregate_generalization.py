"""Pool BER or BLER CSVs into a train-profile by test-profile table."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--metric", choices=["ber", "bler"], required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    files = sorted(input_root.rglob("*.csv"))
    groups = defaultdict(lambda: [0, 0, 0])
    for path in files:
        if path.resolve() == Path(args.out_csv).resolve():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if args.metric == "ber" and "bit_errors" in row:
                    axis_name, numerator, denominator = (
                        "snr_db",
                        "bit_errors",
                        "valid_bits",
                    )
                elif args.metric == "bler" and "block_errors" in row:
                    axis_name, numerator, denominator = (
                        "ebno_db",
                        "block_errors",
                        "num_blocks",
                    )
                else:
                    continue
                key = (
                    row["model"],
                    row["train_profile"],
                    row["test_profile"],
                    row[axis_name],
                )
                groups[key][0] += int(row[numerator])
                groups[key][1] += int(row[denominator])
                groups[key][2] += 1

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    axis_name = "snr_db" if args.metric == "ber" else "ebno_db"
    with out_path.open("w", newline="") as handle:
        fields = [
            "model",
            "train_profile",
            "test_profile",
            axis_name,
            args.metric,
            "errors",
            "trials",
            "num_rows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups, key=lambda k: (k[:3], float(k[3]))):
            errors, trials, num_rows = groups[key]
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [*key, errors / trials, errors, trials, num_rows],
                    )
                )
            )
    print(f"Aggregated {len(files)} CSV files into {out_path}")


if __name__ == "__main__":
    main()
