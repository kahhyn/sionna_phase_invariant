"""Print a parameter summary using the same model builder as training."""

import argparse

from models.factory import MODEL_CHOICES, build_model_from_args
from utils.count_model_params import print_summary, summarize_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hidden_complex", type=int, default=16)
    parser.add_argument("--zero_complex", type=int, default=16)
    parser.add_argument("--branch_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--gate_type", choices=["sigmoid", "swiglu"], default="swiglu")
    parser.add_argument("--single_readout_mode", choices=["low_rank", "full"], default="low_rank")
    parser.add_argument("--zero_gate_hidden", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = build_model_from_args(args, bits_per_symbol=2)
    summary = summarize_model(model, verbose=False)
    print_summary(summary, show_layers=args.verbose)


if __name__ == "__main__":
    main()
