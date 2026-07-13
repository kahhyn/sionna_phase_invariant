"""Model construction utilities shared by training and evaluation scripts."""

from .baseline_cnn import RealImagCNN, PhysicalFeatureCNN
from .complex_no_interaction_cnn import (
    ComplexCNNNoInteraction,
    ComplexCNNWithZeroConditioning,
    ComplexCNNWithZeroInput,
)
from .phase_invariant_net import PhaseInvariantReceiver
from .single_invariant_net import (
    MatchedN0GatedComplexCNN,
    N0GatedSingleBranchPhaseInvariantReceiver,
    SingleBranchPhaseInvariantReceiver,
    StrictMatchedN0GatedComplexCNN,
)


MODEL_CHOICES = (
    "real_imag_cnn",
    "physical_cnn",
    "phase_invariant",
    "complex_no_interaction",
    "complex_p",
    "complex_n0",
    "complex_p_n0",
    "complex_p_n0_gate",
    "complex_p_n0_film",
    "single_branch",
    "single_branch_n0_gate",
    "single_branch_p_only_gate",
    "single_branch_n0_only_gate",
    "matched_complex_p_n0_gate",
    "matched_complex_p_only_gate",
    "matched_complex_n0_only_gate",
    "strict_matched_complex_p_n0_gate",
)

GATE_CONDITION_BY_MODEL = {
    "single_branch_n0_gate": "p_n0",
    "single_branch_p_only_gate": "p_only",
    "single_branch_n0_only_gate": "n0_only",
}

MATCHED_GATE_CONDITION_BY_MODEL = {
    "matched_complex_p_n0_gate": "p_n0",
    "matched_complex_p_only_gate": "p_only",
    "matched_complex_n0_only_gate": "n0_only",
}

COMPLEX_INPUT_CONDITION_BY_MODEL = {
    "complex_p": "p",
    "complex_n0": "n0",
    "complex_p_n0": "p_n0",
}

COMPLEX_CONDITION_METHOD_BY_MODEL = {
    "complex_p_n0_gate": "gate",
    "complex_p_n0_film": "film",
}


def build_model(
    name,
    bits_per_symbol,
    hidden=32,
    hidden_complex=16,
    zero_complex=16,
    branch_layers=2,
    kernel_size=3,
    use_norm=True,
    gate_type="swiglu",
    single_readout_mode="low_rank",
    zero_gate_hidden=16,
):
    """Build a receiver by its public experiment alias."""
    if name == "real_imag_cnn":
        return RealImagCNN(hidden=hidden, bits_per_symbol=bits_per_symbol)
    if name == "physical_cnn":
        return PhysicalFeatureCNN(
            hidden=hidden,
            zero_complex=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
        )
    if name == "phase_invariant":
        return PhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_complex=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )
    if name == "complex_no_interaction":
        return ComplexCNNNoInteraction(
            hidden_complex=hidden_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )
    if name in COMPLEX_INPUT_CONDITION_BY_MODEL:
        return ComplexCNNWithZeroInput(
            condition_mode=COMPLEX_INPUT_CONDITION_BY_MODEL[name],
            hidden_complex=hidden_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )
    if name in COMPLEX_CONDITION_METHOD_BY_MODEL:
        return ComplexCNNWithZeroConditioning(
            condition_mode="p_n0",
            condition_method=COMPLEX_CONDITION_METHOD_BY_MODEL[name],
            hidden_complex=hidden_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            condition_hidden=zero_gate_hidden,
        )
    if name == "single_branch":
        return SingleBranchPhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_real=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            num_blocks=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            readout_mode=single_readout_mode,
        )
    if name in MATCHED_GATE_CONDITION_BY_MODEL:
        return MatchedN0GatedComplexCNN(
            hidden_complex=hidden_complex,
            zero_real=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            num_blocks=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            zero_gate_hidden=zero_gate_hidden,
            zero_gate_condition=MATCHED_GATE_CONDITION_BY_MODEL[name],
        )
    if name == "strict_matched_complex_p_n0_gate":
        return StrictMatchedN0GatedComplexCNN(
            hidden_complex=hidden_complex,
            zero_real=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            num_blocks=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            zero_gate_hidden=zero_gate_hidden,
            zero_gate_condition="p_n0",
        )
    if name in GATE_CONDITION_BY_MODEL:
        return N0GatedSingleBranchPhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_real=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            num_blocks=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            readout_mode=single_readout_mode,
            zero_gate_hidden=zero_gate_hidden,
            zero_gate_condition=GATE_CONDITION_BY_MODEL[name],
        )
    raise ValueError(f"Unknown model: {name}")


def build_model_from_args(args, bits_per_symbol=2):
    """Build a model from an argparse namespace or a checkpoint args dict."""
    if not isinstance(args, dict):
        args = vars(args)
    return build_model(
        args["model"],
        bits_per_symbol=bits_per_symbol,
        hidden=args["hidden"],
        hidden_complex=args["hidden_complex"],
        zero_complex=args["zero_complex"],
        branch_layers=args["branch_layers"],
        kernel_size=args["kernel_size"],
        use_norm=not args["no_norm"],
        gate_type=args["gate_type"],
        single_readout_mode=args["single_readout_mode"],
        zero_gate_hidden=args.get("zero_gate_hidden", 16),
    )
