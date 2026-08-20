"""Audit and summarize the matched RealImagCNN four-domain experiment."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


DOMAIN_PROFILE_NAMES = {
    "tdl_a": "tdl_A_10_30_100_mix_normalized",
    "tdl_mix": "tdl_mix_normalized",
    "umi": "umi_normalized",
    "umi_uma_mix": "umi_uma_mix_normalized",
}


def read_raw_csvs(root):
    rows = []
    for path in sorted(Path(root).rglob("eval_*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(
                    {
                        "path": str(path),
                        "model": row["model"],
                        "train_profile": row["train_profile"],
                        "test_profile": row["test_profile"],
                        "train_seed": int(row["train_seed"]),
                        "eval_seed": int(row["eval_seed"]),
                        "snr_db": float(row["snr_db"]),
                        "bce": float(row["bce"]),
                        "errors": int(row["bit_errors"]),
                        "trials": int(row["valid_bits"]),
                        "common_random_numbers": int(
                            row["common_random_numbers"]
                        ),
                    }
                )
    return rows


def profile_groups(test_profile):
    groups = [test_profile]
    if test_profile.startswith("tdl_"):
        groups.append("tdl_all")
        groups.append(test_profile.split("_", 2)[1].lower().join(["tdl_", ""]))
    elif test_profile == "umi_normalized":
        groups.extend(["umi", "urban_all"])
    elif test_profile == "uma_normalized":
        groups.extend(["uma", "urban_all"])
    return groups


def in_domain(test_profile, domain):
    if domain == "tdl_a":
        return test_profile in {
            "tdl_A_10ns",
            "tdl_A_30ns",
            "tdl_A_100ns",
        }
    if domain == "tdl_mix":
        return test_profile.startswith("tdl_")
    if domain == "umi":
        return test_profile == "umi_normalized"
    if domain == "umi_uma_mix":
        return test_profile in {"umi_normalized", "uma_normalized"}
    raise ValueError(f"Unknown domain: {domain}")


def pooled_by_seed(rows, domain):
    totals = defaultdict(lambda: [0, 0, 0.0, 0])
    for row in rows:
        groups = profile_groups(row["test_profile"])
        if in_domain(row["test_profile"], domain):
            groups.append("in_domain")
        for group in groups:
            key = (row["train_seed"], group, row["snr_db"])
            totals[key][0] += row["errors"]
            totals[key][1] += row["trials"]
            totals[key][2] += row["bce"] * row["trials"]
            totals[key][3] += 1
    result = {}
    for key, (errors, trials, bce_sum, count) in totals.items():
        result[key] = {
            "errors": errors,
            "trials": trials,
            "ber": errors / trials,
            "bce": bce_sum / trials,
            "rows": count,
        }
    return result


def mean_std_ci(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan, math.nan, math.nan
    std = statistics.stdev(values)
    # Formal comparisons use three independent training seeds.
    t_975 = 4.302652729911275 if len(values) == 3 else 1.96
    half = t_975 * std / math.sqrt(len(values))
    return mean, std, mean - half, mean + half


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_domain(domain, rows):
    by_seed = pooled_by_seed(rows, domain)
    grouped = defaultdict(list)
    for (seed, group, snr), value in by_seed.items():
        grouped[(group, snr)].append((seed, value))

    output = []
    for (group, snr), values in sorted(grouped.items()):
        bers = [value["ber"] for _, value in values]
        bces = [value["bce"] for _, value in values]
        mean_ber, std_ber, ci_low, ci_high = mean_std_ci(bers)
        output.append(
            {
                "train_domain": domain,
                "train_profile": DOMAIN_PROFILE_NAMES[domain],
                "test_group": group,
                "snr_db": snr,
                "num_train_seeds": len(values),
                "mean_seed_ber": mean_ber,
                "std_seed_ber": std_ber,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_seed_bce": statistics.fmean(bces),
                "pooled_ber": sum(v["errors"] for _, v in values)
                / sum(v["trials"] for _, v in values),
                "errors": sum(v["errors"] for _, v in values),
                "trials": sum(v["trials"] for _, v in values),
            }
        )
    return output, by_seed


def compare_to_a(domain, real_rows, a_rows):
    real = pooled_by_seed(real_rows, domain)
    baseline = pooled_by_seed(a_rows, domain)
    if set(real) != set(baseline):
        missing_real = sorted(set(baseline) - set(real))[:5]
        missing_a = sorted(set(real) - set(baseline))[:5]
        raise RuntimeError(
            f"Unpaired comparison for {domain}: "
            f"missing real={missing_real}, missing A={missing_a}"
        )

    paired = defaultdict(list)
    for seed, group, snr in sorted(real):
        r = real[(seed, group, snr)]
        a = baseline[(seed, group, snr)]
        paired[(group, snr)].append(
            {
                "seed": seed,
                "real_ber": r["ber"],
                "a_ber": a["ber"],
                "difference": r["ber"] - a["ber"],
                "real_bce": r["bce"],
                "a_bce": a["bce"],
                "bce_difference": r["bce"] - a["bce"],
                "real_errors": r["errors"],
                "real_trials": r["trials"],
                "a_errors": a["errors"],
                "a_trials": a["trials"],
            }
        )

    output = []
    for (group, snr), values in sorted(paired.items()):
        differences = [value["difference"] for value in values]
        bce_differences = [value["bce_difference"] for value in values]
        mean_diff, std_diff, ci_low, ci_high = mean_std_ci(differences)
        mean_bce_diff, std_bce_diff, bce_ci_low, bce_ci_high = mean_std_ci(
            bce_differences
        )
        real_pooled = sum(value["real_errors"] for value in values) / sum(
            value["real_trials"] for value in values
        )
        a_pooled = sum(value["a_errors"] for value in values) / sum(
            value["a_trials"] for value in values
        )
        output.append(
            {
                "train_domain": domain,
                "test_group": group,
                "snr_db": snr,
                "num_train_seeds": len(values),
                "real_pooled_ber": real_pooled,
                "a_pooled_ber": a_pooled,
                "pooled_difference": real_pooled - a_pooled,
                "pooled_ratio": real_pooled / a_pooled if a_pooled else math.inf,
                "mean_paired_seed_difference": mean_diff,
                "std_paired_seed_difference": std_diff,
                "paired_ci95_low": ci_low,
                "paired_ci95_high": ci_high,
                "real_mean_seed_bce": statistics.fmean(
                    value["real_bce"] for value in values
                ),
                "a_mean_seed_bce": statistics.fmean(
                    value["a_bce"] for value in values
                ),
                "mean_paired_seed_bce_difference": mean_bce_diff,
                "std_paired_seed_bce_difference": std_bce_diff,
                "paired_bce_ci95_low": bce_ci_low,
                "paired_bce_ci95_high": bce_ci_high,
            }
        )
    return output


def checkpoint_summary(real_root):
    output = []
    for domain in DOMAIN_PROFILE_NAMES:
        root = real_root / domain / "checkpoints" / "real_imag_cnn"
        for history_path in sorted(root.glob("seed_*/history.csv")):
            with history_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            numeric = [
                {
                    "epoch": int(row["epoch"]),
                    "val_loss": float(row["val_loss"]),
                    "val_ber": float(row["val_ber"]),
                }
                for row in rows
            ]
            best = min(numeric, key=lambda row: row["val_loss"])
            final = numeric[-1]
            output.append(
                {
                    "train_domain": domain,
                    "train_profile": DOMAIN_PROFILE_NAMES[domain],
                    "train_seed": int(history_path.parent.name.split("_")[-1]),
                    "epochs": len(numeric),
                    "best_epoch": best["epoch"],
                    "best_val_bce": best["val_loss"],
                    "best_val_ber": best["val_ber"],
                    "final_val_bce": final["val_loss"],
                    "final_val_ber": final["val_ber"],
                    "best_checkpoint_exists": int(
                        (history_path.parent / "best.pt").is_file()
                    ),
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real_root",
        type=Path,
        default=Path("runs/realimag_matched_four_domains_formal"),
    )
    parser.add_argument(
        "--a_tdl_root",
        type=Path,
        default=Path(
            "runs/pilot_tdl_mix_to_urban/eval_ber/single_branch_n0_gate"
        ),
    )
    parser.add_argument(
        "--a_urban_root",
        type=Path,
        default=Path(
            "runs/pilot_urban_mix_to_tdl/eval_ber/single_branch_n0_gate"
        ),
    )
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.real_root / "analysis"

    checkpoint_rows = checkpoint_summary(args.real_root)
    if len(checkpoint_rows) != 12:
        raise RuntimeError(f"Expected 12 checkpoint histories, got {len(checkpoint_rows)}")
    write_csv(
        out_dir / "checkpoint_summary.csv",
        checkpoint_rows,
        list(checkpoint_rows[0]),
    )

    all_domain_rows = []
    real_by_domain = {}
    for domain in DOMAIN_PROFILE_NAMES:
        rows = read_raw_csvs(args.real_root / domain / "eval_ber")
        if len(rows) != 132 * 16:
            raise RuntimeError(
                f"Expected 2112 raw rows for {domain}, got {len(rows)}"
            )
        if any(row["common_random_numbers"] != 1 for row in rows):
            raise RuntimeError(f"CRN disabled in one or more {domain} rows")
        if any(not math.isfinite(row["bce"]) for row in rows):
            raise RuntimeError(f"Non-finite BCE in {domain}")
        summary, _ = summarize_domain(domain, rows)
        all_domain_rows.extend(summary)
        real_by_domain[domain] = rows
    write_csv(
        out_dir / "realimag_domain_summary.csv",
        all_domain_rows,
        list(all_domain_rows[0]),
    )

    comparison_rows = []
    for domain, a_root in (
        ("tdl_mix", args.a_tdl_root),
        ("umi_uma_mix", args.a_urban_root),
    ):
        a_rows = read_raw_csvs(a_root)
        if len(a_rows) != 132 * 16:
            raise RuntimeError(
                f"Expected 2112 A rows for {domain}, got {len(a_rows)}"
            )
        comparison_rows.extend(
            compare_to_a(domain, real_by_domain[domain], a_rows)
        )
    write_csv(
        out_dir / "realimag_vs_single_branch.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )

    print(f"Wrote audited summaries to {out_dir}")


if __name__ == "__main__":
    main()
