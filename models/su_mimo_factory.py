"""Model construction shared by SU-MIMO training and checkpoint loading."""

from .su_mimo_invariant_net import (
    SUMIMOCanonicalPhaseReceiver,
    SUMIMOPhaseInvariantReceiver,
    SUMIMOPhaseSensitiveReceiver,
)
from .su_mimo_real_cnn import SUMIMORealCNNReceiver


SU_MIMO_MODEL_CHOICES = (
    "su_mimo_phase_invariant",
    "su_mimo_phase_canonical",
    "su_mimo_phase_sensitive",
    "su_mimo_real_cnn",
)


def build_su_mimo_model(name, model_config):
    """Build a layer-equivariant SU-MIMO receiver by experiment alias."""
    if name == "su_mimo_phase_invariant":
        return SUMIMOPhaseInvariantReceiver(**model_config)
    if name == "su_mimo_phase_canonical":
        return SUMIMOCanonicalPhaseReceiver(**model_config)
    if name == "su_mimo_phase_sensitive":
        return SUMIMOPhaseSensitiveReceiver(**model_config)
    if name == "su_mimo_real_cnn":
        reference = SUMIMOPhaseSensitiveReceiver(**model_config)
        target_parameter_count = sum(
            parameter.numel() for parameter in reference.parameters()
        )
        return SUMIMORealCNNReceiver(
            **model_config,
            target_parameter_count=target_parameter_count,
        )
    raise ValueError(f"Unknown SU-MIMO model: {name}")
