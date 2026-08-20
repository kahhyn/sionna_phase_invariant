"""Aggregate and validate the frozen seven-configuration SU-MIMO LMMSE matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPATIAL_CONFIGS = {
    "layer2_rx2": (2, 2),
    "layer2_rx4": (2, 4),
    "layer2_rx8": (2, 8),
    "layer2_rx16": (2, 16),
    "layer4_rx4": (4, 4),
    "layer4_rx8": (4, 8),
    "layer4_rx16": (4, 16),
}
PROFILES = ("umi_normalized", "uma_normalized")
RECEIVERS = ("lmmse_ls", "lmmse_perfect")
EBNO_VALUES = (3.0, 5.0, 7.0, 9.0, 11.0, 13.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(run_root: Path, require_complete: bool) -> tuple[int, int]:
    aggregate_rows: list[dict[str, str]] = []
    layer_rows: list[dict[str, str]] = []
    missing: list[str] = []
    invalid: list[str] = []

    for spatial_id, (num_layers, num_rx_ant) in SPATIAL_CONFIGS.items():
        for profile in PROFILES:
            for receiver in RECEIVERS:
                result_dir = run_root / spatial_id / profile / receiver
                aggregate_path = result_dir / "bler.csv"
                layer_path = result_dir / "bler_per_layer.csv"
                if not aggregate_path.exists() or not layer_path.exists():
                    missing.append(f"{spatial_id}/{profile}/{receiver}")
                    continue

                rows = read_rows(aggregate_path)
                observed_ebno = tuple(float(row["ebno_db"]) for row in rows)
                if observed_ebno != EBNO_VALUES:
                    invalid.append(
                        f"{aggregate_path}: Eb/N0 {observed_ebno}, expected {EBNO_VALUES}"
                    )
                    continue
                for row in rows:
                    if row["receiver"] != receiver or row["test_profile"] != profile:
                        invalid.append(f"{aggregate_path}: metadata mismatch")
                        break
                    if (
                        int(row["num_layers"]) != num_layers
                        or int(row["num_rx_ant"]) != num_rx_ant
                    ):
                        invalid.append(f"{aggregate_path}: spatial metadata mismatch")
                        break
                    aggregate_rows.append(
                        {"spatial_id": spatial_id, "source_csv": str(aggregate_path), **row}
                    )

                expected_layers = num_layers
                per_layer = read_rows(layer_path)
                if len(per_layer) != len(EBNO_VALUES) * expected_layers:
                    invalid.append(
                        f"{layer_path}: {len(per_layer)} rows, expected "
                        f"{len(EBNO_VALUES) * expected_layers}"
                    )
                    continue
                layer_rows.extend(
                    {"spatial_id": spatial_id, "source_csv": str(layer_path), **row}
                    for row in per_layer
                )

    if invalid:
        raise ValueError("Invalid result files:\n" + "\n".join(invalid))
    if require_complete and missing:
        raise FileNotFoundError("Missing result cells:\n" + "\n".join(missing))
    if not aggregate_rows:
        raise FileNotFoundError(f"No completed result files under {run_root}.")

    write_rows(run_root / "aggregate_bler.csv", aggregate_rows)
    write_rows(run_root / "aggregate_per_layer.csv", layer_rows)
    if require_complete and len(aggregate_rows) != 168:
        raise ValueError(f"Expected 168 aggregate rows, found {len(aggregate_rows)}.")
    return len(aggregate_rows), len(layer_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    aggregate_count, layer_count = aggregate(args.run_root, args.require_complete)
    print(
        f"Wrote {aggregate_count} aggregate rows and {layer_count} per-layer rows "
        f"under {args.run_root}."
    )


if __name__ == "__main__":
    main()
