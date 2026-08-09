"""Train missing checkpoints and execute the SISO/SU-MIMO OOD matrix.

The runner is intentionally a thin, auditable orchestrator around the existing
training and evaluation entrypoints. ``--mode audit --dry-run`` materializes
the full command plan without importing Sionna or requiring a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path(
    "configs/experiment_matrices/phase_invariance_siso_mimo_v1.json"
)
TRAIN_PROFILES = {
    "tdl_a": Path("configs/channel_profiles/tdl_a_10ns_normalized.json"),
    "tdl_mix": Path("configs/channel_profiles/tdl_mix_normalized.json"),
}
TEST_PROFILES = {
    "delay_ood": Path("configs/channel_profiles/tdl_a_600_1000_ood_normalized.json"),
    "doppler_ood": Path(
        "configs/channel_profiles/tdl_a_10ns_doppler_ood_normalized.json"
    ),
}
DOPPLER_COMPONENTS = (
    "tdl_A_10ns_fd0",
    "tdl_A_10ns_fd400",
    "tdl_A_10ns_fd800",
    "tdl_A_10ns_fd1200",
)
SYSTEMS = ("siso_1l1rx", "mimo_2l2rx", "mimo_2l16rx")
TRAIN_DOMAINS = ("tdl_a", "tdl_mix")
TEST_DOMAINS = (
    "id_fixed",
    "phase_ood",
    "delay_ood",
    "doppler_ood",
    "quadriga_ood",
)


@dataclass(frozen=True)
class Method:
    key: str
    model_name: str


SISO_METHODS = (
    Method("invariant", "single_branch_n0_gate"),
    Method("sensitive", "strict_matched_complex_p_n0_gate"),
)
MIMO_METHODS = (
    Method("invariant", "su_mimo_phase_canonical"),
    Method("sensitive", "su_mimo_phase_sensitive"),
)


def parse_csv_values(text: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in text.replace(" ", ",").split(",") if value.strip())


def parse_int_values(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_csv_values(text))


def run_path(args, *parts: str) -> Path:
    return args.run_root.joinpath(*parts)


def abs_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def tracked_checkpoint(system: str, train_domain: str, method: Method, seed: int) -> Path | None:
    if system == "siso_1l1rx":
        if train_domain == "tdl_a":
            return Path(
                f"checkpoints/continual_source/{method.model_name}_seed{seed}.pt"
            )
        return Path(
            "checkpoints/generalization/tdl_mix_normalized/"
            f"{method.model_name}_seed{seed}.pt"
        )
    if train_domain == "tdl_mix" and seed == 0:
        rx = 2 if system == "mimo_2l2rx" else 16
        return Path(
            "checkpoints/phase_invariance_matrix_v1/mimo_tdl_mix/"
            f"{method.model_name}_rx{rx}_seed0.pt"
        )
    return None


def trained_checkpoint(args, system: str, train_domain: str, method: Method, seed: int) -> Path:
    tracked = tracked_checkpoint(system, train_domain, method, seed)
    if tracked is not None and abs_repo(tracked).exists():
        return abs_repo(tracked)
    return run_path(
        args,
        "training",
        system,
        train_domain,
        method.key,
        f"seed_{seed}",
        "best.pt",
    )


def selected_methods(system: str) -> tuple[Method, ...]:
    return SISO_METHODS if system == "siso_1l1rx" else MIMO_METHODS


def build_mimo_train_command(args, system: str, train_domain: str, method: Method, seed: int) -> list[str]:
    if system == "siso_1l1rx":
        raise ValueError("The matrix runner does not retrain curated SISO checkpoints.")
    rx = 2 if system == "mimo_2l2rx" else 16
    save_dir = trained_checkpoint(args, system, train_domain, method, seed).parent
    profile = TRAIN_PROFILES[train_domain]
    return [
        args.python,
        "-m",
        "training.train_su_mimo",
        "--model",
        method.model_name,
        "--num_layers",
        "2",
        "--num_rx_ant",
        str(rx),
        "--total_tx_power",
        "1.0",
        "--train_channel_profile",
        str(profile),
        "--val_channel_profile",
        str(profile),
        "--train_phase_mode",
        "fixed",
        "--val_phase_mode",
        "fixed",
        "--snr_db_min",
        "-5",
        "--snr_db_max",
        "20",
        "--num_train",
        str(args.num_train),
        "--num_val",
        str(args.num_val),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.train_batch_size),
        "--hidden_complex",
        "32",
        "--zero_real",
        "22",
        "--hidden_real",
        "66",
        "--num_iterations",
        "2",
        "--kernel_size",
        "3",
        "--zero_gate_hidden",
        "16",
        "--lr",
        "1e-3",
        "--lr_scheduler",
        "cosine",
        "--lr_min",
        "1e-5",
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--constant_tail_epochs",
        str(args.constant_tail_epochs),
        "--weight_decay",
        "0",
        "--seed",
        str(seed),
        "--train_generator_seed",
        str(seed),
        "--val_generator_seed",
        str(seed + 100000),
        "--deterministic_algorithms",
        "--device",
        args.device,
        "--save_dir",
        str(save_dir),
    ]


def test_variants(train_domain: str, test_domain: str):
    if test_domain == "id_fixed":
        return [("aggregate", TRAIN_PROFILES[train_domain], "fixed", None)]
    if test_domain == "phase_ood":
        return [("aggregate", TRAIN_PROFILES[train_domain], "uniform", None)]
    if test_domain == "delay_ood":
        return [("aggregate", TEST_PROFILES[test_domain], "uniform", None)]
    if test_domain == "doppler_ood":
        return [
            (component, TEST_PROFILES[test_domain], "uniform", component)
            for component in DOPPLER_COMPONENTS
        ]
    if test_domain == "quadriga_ood":
        return []
    raise ValueError(test_domain)


def build_standard_eval_command(
    args,
    system: str,
    checkpoint: Path,
    metric: str,
    profile: Path,
    phase_mode: str,
    component: str | None,
    eval_seed: int,
    out_csv: Path,
) -> list[str]:
    is_siso = system == "siso_1l1rx"
    if metric == "ber":
        module = "evaluation.eval_ber_sionna" if is_siso else "evaluation.eval_ber_su_mimo"
        command = [
            args.python,
            "-m",
            module,
            "--checkpoint",
            str(checkpoint),
            "--phase_mode",
            phase_mode,
            f"--snr_list={args.snr_list}",
            "--num_samples",
            str(args.num_samples),
            "--batch_size",
            str(args.eval_batch_size),
            "--seed",
            str(eval_seed),
            "--common_random_numbers",
            "--device",
            args.device,
            "--eval_channel_profile",
            str(profile),
            "--out_csv",
            str(out_csv),
        ]
    else:
        module = "evaluation.eval_bler_sionna" if is_siso else "evaluation.eval_bler_su_mimo"
        command = [
            args.python,
            "-m",
            module,
            "--checkpoint",
            str(checkpoint),
            "--phase_mode",
            phase_mode,
            f"--ebno_list={args.ebno_list}",
            "--coderate",
            str(args.coderate),
            "--decoder_iterations",
            str(args.decoder_iterations),
            "--batch_size",
            str(args.eval_batch_size),
            "--target_block_errors",
            str(args.target_block_errors),
            "--max_blocks",
            str(args.max_blocks),
            "--seed",
            str(eval_seed),
            "--common_random_numbers",
            "--device",
            args.device,
            "--eval_channel_profile",
            str(profile),
            "--out_csv",
            str(out_csv),
        ]
    if component:
        command.extend(["--eval_component_id", component])
    return command


def quadriga_dir(args, system: str) -> Path | None:
    value = {
        "siso_1l1rx": args.quadriga_siso_dir,
        "mimo_2l2rx": args.quadriga_mimo_rx2_dir,
        "mimo_2l16rx": args.quadriga_mimo_rx16_dir,
    }[system]
    return None if value is None else Path(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def command_text(command: Iterable[str]) -> str:
    return shlex.join(str(value) for value in command)


def execute_command(args, command: list[str], output_file: Path | None, metadata: dict, records: list[dict]) -> None:
    record = {**metadata, "command": command_text(command), "output_file": str(output_file or "")}
    if output_file is not None and args.skip_existing and output_file.exists():
        record["status"] = "skipped_existing"
        records.append(record)
        print(f"SKIP {output_file}")
        return
    print(command_text(command))
    if args.dry_run:
        record["status"] = "planned"
        records.append(record)
        return
    if output_file is not None:
        write_json(output_file.parent / "cell_manifest.json", record)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=os.environ.copy())
    record["status"] = "completed"
    records.append(record)


def audit_checkpoints(args) -> list[dict]:
    rows = []
    for system in args.systems:
        for train_domain in args.train_domains:
            for method in selected_methods(system):
                for seed in args.seeds:
                    path = trained_checkpoint(args, system, train_domain, method, seed)
                    rows.append(
                        {
                            "system": system,
                            "train_domain": train_domain,
                            "method": method.key,
                            "model_name": method.model_name,
                            "seed": seed,
                            "checkpoint": str(path),
                            "exists": path.exists(),
                            "needs_training": not path.exists(),
                            "trainable_by_runner": system != "siso_1l1rx",
                        }
                    )
    return rows


def train_missing(args, records: list[dict]) -> None:
    for row in audit_checkpoints(args):
        if not row["needs_training"]:
            continue
        if not row["trainable_by_runner"]:
            raise FileNotFoundError(
                f"Missing curated SISO checkpoint: {row['checkpoint']}"
            )
        method = next(
            method
            for method in MIMO_METHODS
            if method.key == row["method"]
        )
        command = build_mimo_train_command(
            args,
            row["system"],
            row["train_domain"],
            method,
            row["seed"],
        )
        checkpoint = Path(row["checkpoint"])
        execute_command(
            args,
            command,
            checkpoint,
            {"stage": "train", **row},
            records,
        )


def evaluate_standard(args, records: list[dict]) -> None:
    for system in args.systems:
        for train_domain in args.train_domains:
            for method in selected_methods(system):
                for seed in args.seeds:
                    checkpoint = trained_checkpoint(args, system, train_domain, method, seed)
                    if not checkpoint.exists() and not args.dry_run:
                        raise FileNotFoundError(checkpoint)
                    for test_domain in args.test_domains:
                        if test_domain == "quadriga_ood":
                            continue
                        for variant, profile, phase_mode, component in test_variants(
                            train_domain, test_domain
                        ):
                            for eval_seed in args.eval_seeds:
                                for metric in args.metrics:
                                    output_dir = run_path(
                                        args,
                                        "evaluation",
                                        system,
                                        train_domain,
                                        test_domain,
                                        variant,
                                        method.key,
                                        f"train_seed_{seed}",
                                        f"eval_seed_{eval_seed}",
                                    )
                                    out_csv = output_dir / f"{metric}.csv"
                                    command = build_standard_eval_command(
                                        args,
                                        system,
                                        checkpoint,
                                        metric,
                                        profile,
                                        phase_mode,
                                        component,
                                        eval_seed,
                                        out_csv,
                                    )
                                    execute_command(
                                        args,
                                        command,
                                        out_csv,
                                        {
                                            "stage": "evaluate",
                                            "system": system,
                                            "train_domain": train_domain,
                                            "test_domain": test_domain,
                                            "variant": variant,
                                            "method": method.key,
                                            "model_name": method.model_name,
                                            "train_seed": seed,
                                            "eval_seed": eval_seed,
                                            "metric": metric,
                                            "phase_mode": phase_mode,
                                            "profile": str(profile),
                                        },
                                        records,
                                    )


def evaluate_quadriga(args, records: list[dict]) -> None:
    if "quadriga_ood" not in args.test_domains or "bler" not in args.metrics:
        return
    for system in args.systems:
        directory = quadriga_dir(args, system)
        mats = [] if directory is None else sorted(directory.glob(args.quadriga_pattern))
        if not mats:
            message = f"No QuaDRiGa MAT files for {system}: {directory}"
            if args.require_quadriga and not args.dry_run:
                raise FileNotFoundError(message)
            print(f"SKIP {message}")
            records.append(
                {
                    "stage": "evaluate",
                    "system": system,
                    "test_domain": "quadriga_ood",
                    "status": "missing_external_data",
                    "directory": str(directory or ""),
                }
            )
            continue
        for train_domain in args.train_domains:
            methods = selected_methods(system)
            for seed in args.seeds:
                invariant = trained_checkpoint(args, system, train_domain, methods[0], seed)
                sensitive = trained_checkpoint(args, system, train_domain, methods[1], seed)
                for mat_path in mats:
                    trajectory = mat_path.stem
                    for eval_seed in args.eval_seeds:
                        output_dir = run_path(
                            args,
                            "evaluation",
                            system,
                            train_domain,
                            "quadriga_ood",
                            trajectory,
                            f"train_seed_{seed}",
                            f"eval_seed_{eval_seed}",
                        )
                        out_csv = output_dir / "quadriga_bler_summary.csv"
                        command = [
                            args.python,
                            "-m",
                            "evaluation.eval_quadriga_bler_matrix",
                            "--system",
                            "siso" if system == "siso_1l1rx" else "su_mimo",
                            "--channel_mat",
                            str(mat_path),
                            "--invariant_checkpoint",
                            str(invariant),
                            "--sensitive_checkpoint",
                            str(sensitive),
                            "--output_dir",
                            str(output_dir),
                            f"--ebno_list={args.ebno_list}",
                            "--coderate",
                            str(args.coderate),
                            "--decoder_iterations",
                            str(args.decoder_iterations),
                            "--phase_mode",
                            "uniform",
                            "--batch_size",
                            str(args.eval_batch_size),
                            "--repetitions",
                            str(args.quadriga_repetitions),
                            "--seed",
                            str(eval_seed),
                            "--normalization",
                            "checkpoint",
                            "--mimo_mat_layout",
                            args.mimo_mat_layout,
                            "--device",
                            args.device,
                        ]
                        execute_command(
                            args,
                            command,
                            out_csv,
                            {
                                "stage": "evaluate",
                                "system": system,
                                "train_domain": train_domain,
                                "test_domain": "quadriga_ood",
                                "variant": trajectory,
                                "method": "paired",
                                "train_seed": seed,
                                "eval_seed": eval_seed,
                                "metric": "bler",
                                "phase_mode": "uniform",
                                "channel_mat": str(mat_path),
                            },
                            records,
                        )


def aggregate_results(args) -> None:
    aggregate_root = run_path(args, "aggregate")
    aggregate_root.mkdir(parents=True, exist_ok=True)
    groups = {
        "ber": ("ber.csv",),
        "bler": ("bler.csv", "quadriga_bler_summary.csv"),
        "per_layer": ("ber_per_layer.csv", "bler_per_layer.csv", "quadriga_bler_per_layer.csv"),
    }
    for group, names in groups.items():
        rows = []
        for path in run_path(args, "evaluation").rglob("*.csv"):
            if path.name not in names:
                continue
            manifest_path = path.parent / "cell_manifest.json"
            metadata = {}
            if manifest_path.exists():
                with manifest_path.open() as stream:
                    metadata = json.load(stream)
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    rows.append(
                        {
                            "matrix_system": metadata.get("system", ""),
                            "matrix_train_domain": metadata.get("train_domain", ""),
                            "matrix_test_domain": metadata.get("test_domain", ""),
                            "matrix_variant": metadata.get("variant", ""),
                            "matrix_method": metadata.get("method", ""),
                            "matrix_metric": metadata.get("metric", ""),
                            "source_csv": str(path),
                            **row,
                        }
                    )
        if rows:
            out = aggregate_root / f"matrix_{group}.csv"
            fieldnames = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with out.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {len(rows)} rows to {out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "train", "evaluate", "all", "aggregate"], default="audit")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--run_root", type=Path, default=Path("runs/phase_invariance_matrix_v1"))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--train_domains", default=",".join(TRAIN_DOMAINS))
    parser.add_argument("--test_domains", default=",".join(TEST_DOMAINS))
    parser.add_argument("--metrics", default="bler", help="ber,bler")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--eval_seeds", default="777000")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_quadriga", action="store_true")
    parser.add_argument("--quadriga_siso_dir", type=Path)
    parser.add_argument("--quadriga_mimo_rx2_dir", type=Path)
    parser.add_argument("--quadriga_mimo_rx16_dir", type=Path)
    parser.add_argument("--quadriga_pattern", default="*.mat")
    parser.add_argument("--quadriga_repetitions", type=int, default=1)
    parser.add_argument(
        "--mimo_mat_layout",
        default="frame_rx_layer_symbol_subcarrier",
        choices=["frame_rx_layer_symbol_subcarrier", "frame_layer_rx_symbol_subcarrier"],
    )
    parser.add_argument("--snr_list", default="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20")
    parser.add_argument("--ebno_list", default="-5,-3,-1,0,1,2,3,4,5,6,7,8,9,11,13")
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--target_block_errors", type=int, default=500)
    parser.add_argument("--max_blocks", type=int, default=20000)
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--decoder_iterations", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--constant_tail_epochs", type=int, default=30)
    args = parser.parse_args(argv)
    args.run_root = abs_repo(args.run_root)
    args.contract = abs_repo(args.contract)
    args.systems = parse_csv_values(args.systems)
    args.train_domains = parse_csv_values(args.train_domains)
    args.test_domains = parse_csv_values(args.test_domains)
    args.metrics = parse_csv_values(args.metrics)
    args.seeds = parse_int_values(args.seeds)
    args.eval_seeds = parse_int_values(args.eval_seeds)
    for values, allowed, name in (
        (args.systems, SYSTEMS, "systems"),
        (args.train_domains, TRAIN_DOMAINS, "train_domains"),
        (args.test_domains, TEST_DOMAINS, "test_domains"),
        (args.metrics, ("ber", "bler"), "metrics"),
    ):
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown {name}: {unknown}")
    return args


def main(argv=None):
    args = parse_args(argv)
    with args.contract.open() as stream:
        contract = json.load(stream)
    records: list[dict] = []
    audit = audit_checkpoints(args)
    print(
        f"Checkpoint audit: {sum(row['exists'] for row in audit)}/{len(audit)} "
        "available"
    )
    for row in audit:
        state = "READY" if row["exists"] else "TRAIN"
        print(
            f"{state:5s} {row['system']:14s} {row['train_domain']:7s} "
            f"{row['method']:9s} seed={row['seed']} {row['checkpoint']}"
        )
    args.run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.run_root / "matrix_plan.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": contract,
            "selection": {
                "systems": args.systems,
                "train_domains": args.train_domains,
                "test_domains": args.test_domains,
                "metrics": args.metrics,
                "seeds": args.seeds,
                "eval_seeds": args.eval_seeds,
            },
            "checkpoint_audit": audit,
        },
    )
    if args.mode in {"train", "all"}:
        train_missing(args, records)
    if args.mode in {"evaluate", "all"}:
        evaluate_standard(args, records)
        evaluate_quadriga(args, records)
        if not args.dry_run:
            aggregate_results(args)
    elif args.mode == "aggregate":
        aggregate_results(args)
    write_json(args.run_root / "execution_records.json", {"records": records})


if __name__ == "__main__":
    main()
