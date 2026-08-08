"""Model construction shared by SU-MIMO training and checkpoint loading."""

from .su_mimo_invariant_net import (
    SUMIMOCanonicalPhaseReceiver,
    SUMIMOPhaseInvariantReceiver,
    SUMIMOPhaseSensitiveReceiver,
)


SU_MIMO_MODEL_CHOICES = (
    "su_mimo_phase_invariant",
    "su_mimo_phase_canonical",
    "su_mimo_phase_sensitive",
)


def build_su_mimo_model(name, model_config):
    """Build a layer-equivariant SU-MIMO receiver by experiment alias."""
    if name == "su_mimo_phase_invariant":
        return SUMIMOPhaseInvariantReceiver(**model_config)
    if name == "su_mimo_phase_canonical":
        return SUMIMOCanonicalPhaseReceiver(**model_config)
    if name == "su_mimo_phase_sensitive":
        return SUMIMOPhaseSensitiveReceiver(**model_config)
    raise ValueError(f"Unknown SU-MIMO model: {name}")
