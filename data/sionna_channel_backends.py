"""Channel-profile loading and Sionna TDL/UMi/UMa backends.

Profiles switch channels at batch granularity.  The OFDM grid, mapper, noise
model, and estimator remain owned by :mod:`data.sionna_ofdm_generator`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sionna.phy.channel import OFDMChannel, gen_single_sector_topology
from sionna.phy.channel.tr38901 import PanelArray, TDL, UMa, UMi


PROFILE_SCHEMA_VERSION = 1
SUPPORTED_BACKENDS = {"tdl", "umi", "uma"}


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def channel_profile_hash(profile: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a validated profile."""
    return hashlib.sha256(_canonical_json(profile).encode("utf-8")).hexdigest()


def legacy_channel_profile(config: Any) -> dict[str, Any]:
    """Translate the historical TDL CLI/config fields into a profile."""
    return validate_channel_profile(
        {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": f"tdl_{config.tdl_model}_{config.delay_spread_s * 1e9:g}ns_legacy",
            "sampling": "balanced_batch",
            "components": [
                {
                    "id": "legacy_tdl",
                    "backend": "tdl",
                    "weight": 1.0,
                    "tdl_model": config.tdl_model,
                    "delay_spread_ns": config.delay_spread_s * 1e9,
                    "max_doppler_hz": config.max_doppler_hz,
                    "normalize_channel": config.normalize_channel,
                }
            ],
        }
    )


def validate_channel_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a channel profile dictionary."""
    if not isinstance(profile, dict):
        raise TypeError("channel profile must be a JSON object")
    result = copy.deepcopy(profile)
    version = result.get("schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported channel profile schema_version={version!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}."
        )
    name = result.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("channel profile requires a non-empty name")
    sampling = result.get("sampling", "balanced_batch")
    if sampling not in {"balanced_batch", "weighted_random"}:
        raise ValueError("sampling must be balanced_batch or weighted_random")
    result["sampling"] = sampling

    components = result.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("channel profile requires at least one component")
    seen_ids: set[str] = set()
    total_weight = 0.0
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise TypeError(f"components[{index}] must be a JSON object")
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError(f"components[{index}] requires a non-empty id")
        if component_id in seen_ids:
            raise ValueError(f"duplicate channel component id: {component_id}")
        seen_ids.add(component_id)
        backend = str(component.get("backend", "")).lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r} in component {component_id!r}."
            )
        component["backend"] = backend
        weight = float(component.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError(f"component {component_id!r} has negative weight")
        component["weight"] = weight
        total_weight += weight

        normalize = component.get("normalize_channel", True)
        if not isinstance(normalize, bool):
            raise TypeError("normalize_channel must be boolean")
        if backend == "tdl":
            model = str(component.get("tdl_model", ""))
            if model not in {"A", "B", "C", "D", "E", "A30", "B100", "C300"}:
                raise ValueError(f"Invalid TDL model {model!r}")
            if float(component.get("delay_spread_ns", 0.0)) <= 0.0:
                raise ValueError("TDL delay_spread_ns must be positive")
        else:
            if component.get("direction", "uplink") != "uplink":
                raise ValueError("The current SISO backend supports uplink only")
            if component.get("o2i_model", "low") not in {"low", "high"}:
                raise ValueError("o2i_model must be low or high")
            indoor_probability = float(component.get("indoor_probability", 0.0))
            if not 0.0 <= indoor_probability <= 1.0:
                raise ValueError("indoor_probability must be in [0, 1]")
            min_speed = float(component.get("min_ut_velocity_mps", 0.0))
            max_speed = float(component.get("max_ut_velocity_mps", 0.0))
            if min_speed < 0.0 or max_speed < min_speed:
                raise ValueError("invalid UT velocity range")
    if total_weight <= 0.0:
        raise ValueError("channel component weights must have a positive sum")
    if sampling == "balanced_batch":
        if any(c["weight"] == 0.0 for c in components):
            raise ValueError("balanced_batch requires positive component weights")
        positive_weights = [c["weight"] for c in components]
        if len(set(positive_weights)) > 1:
            raise ValueError("balanced_batch requires equal positive weights")
    return result


def load_channel_profile(
    source: str | Path | dict[str, Any], component_id: str | None = None
) -> dict[str, Any]:
    """Load a profile from JSON or a dictionary and optionally select one component."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    elif isinstance(source, dict):
        profile = copy.deepcopy(source)
    else:
        raise TypeError("profile source must be a path or dictionary")
    profile = validate_channel_profile(profile)
    if component_id is None:
        return profile
    matches = [c for c in profile["components"] if c["id"] == component_id]
    if not matches:
        raise ValueError(
            f"Component {component_id!r} was not found in profile {profile['name']!r}."
        )
    profile["name"] = component_id
    profile["components"] = matches
    profile["sampling"] = "balanced_batch"
    return validate_channel_profile(profile)


def profile_backend_label(profile: dict[str, Any]) -> str:
    backends = sorted({component["backend"] for component in profile["components"]})
    return backends[0] if len(backends) == 1 else "+".join(backends)


def profile_component(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Return the sole component for fixed profiles, otherwise ``None``."""
    return profile["components"][0] if len(profile["components"]) == 1 else None


class ChannelProfileSampler:
    """Independent batch-level component sampler."""

    def __init__(self, profile: dict[str, Any], seed: int) -> None:
        self.profile = profile
        self.generator = torch.Generator(device="cpu")
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))
        self._order: list[int] = []
        self._cursor = 0

    def next_index(self) -> int:
        components = self.profile["components"]
        if len(components) == 1:
            return 0
        if self.profile["sampling"] == "weighted_random":
            weights = torch.tensor([c["weight"] for c in components], dtype=torch.float64)
            return int(torch.multinomial(weights, 1, generator=self.generator).item())
        if self._cursor >= len(self._order):
            self._order = torch.randperm(
                len(components), generator=self.generator
            ).tolist()
            self._cursor = 0
        index = self._order[self._cursor]
        self._cursor += 1
        return index


class SionnaChannelBackend:
    """Cache Sionna channel objects and apply the selected profile component."""

    def __init__(self, profile, config, resource_grid, device) -> None:
        self.profile = load_channel_profile(profile)
        self.config = config
        self.resource_grid = resource_grid
        self.device = torch.device(device)
        self.num_rx_ant = int(getattr(config, "num_rx_ant", 1))
        self.num_tx_ant = int(getattr(config, "num_layers", 1))
        self._runtimes = [self._build_runtime(c) for c in self.profile["components"]]
        self.sampler = ChannelProfileSampler(self.profile, seed=0)
        self.counts: Counter[str] = Counter()
        self.batch_index = 0

    def _build_runtime(self, component: dict[str, Any]) -> dict[str, Any]:
        backend = component["backend"]
        if backend == "tdl":
            max_doppler_hz = float(
                component.get("max_doppler_hz", self.config.max_doppler_hz)
            )
            max_speed_mps = (
                max_doppler_hz
                * 299_792_458.0
                / self.config.carrier_frequency_hz
            )
            channel_model = TDL(
                model=component["tdl_model"],
                delay_spread=float(component["delay_spread_ns"]) * 1e-9,
                carrier_frequency=self.config.carrier_frequency_hz,
                min_speed=0.0,
                max_speed=max_speed_mps,
                num_rx_ant=self.num_rx_ant,
                num_tx_ant=self.num_tx_ant,
                device=str(self.device),
            )
        else:
            common_array_kwargs = dict(
                num_rows_per_panel=1,
                polarization="single",
                polarization_type="V",
                antenna_pattern="omni",
                carrier_frequency=self.config.carrier_frequency_hz,
                device=str(self.device),
            )
            ut_array = PanelArray(
                num_cols_per_panel=self.num_tx_ant,
                **common_array_kwargs,
            )
            bs_array = PanelArray(
                num_cols_per_panel=self.num_rx_ant,
                **common_array_kwargs,
            )
            model_class = UMi if backend == "umi" else UMa
            channel_model = model_class(
                carrier_frequency=self.config.carrier_frequency_hz,
                o2i_model=component.get("o2i_model", "low"),
                ut_array=ut_array,
                bs_array=bs_array,
                direction=component.get("direction", "uplink"),
                enable_pathloss=bool(component.get("enable_pathloss", False)),
                enable_shadow_fading=bool(
                    component.get("enable_shadow_fading", False)
                ),
                device=str(self.device),
            )
        ofdm_channel = OFDMChannel(
            channel_model=channel_model,
            resource_grid=self.resource_grid,
            normalize_channel=bool(component.get("normalize_channel", True)),
            return_channel=True,
            device=str(self.device),
        )
        return {
            "model": channel_model,
            "channel": ofdm_channel,
            "topology_batch_size": None,
        }

    def reset(self, seed: int) -> None:
        self.sampler.reset(seed)
        self.counts.clear()
        self.batch_index = 0
        self.topology_seed_base = int(seed)

    def reset_sampler(self, seed: int) -> None:
        self.sampler.reset(seed)
        self.counts.clear()

    def _prepare_system_topology(
        self, component: dict[str, Any], runtime: dict[str, Any], batch_size: int
    ) -> dict[str, Any]:
        backend = component["backend"]
        channel_model = runtime["model"]
        previous_batch_size = runtime["topology_batch_size"]
        if previous_batch_size is not None and previous_batch_size != batch_size:
            channel_model.reset_topology()
        topology = gen_single_sector_topology(
            batch_size=batch_size,
            num_ut=1,
            scenario=backend,
            indoor_probability=float(component.get("indoor_probability", 0.0)),
            min_ut_velocity=float(component.get("min_ut_velocity_mps", 0.0)),
            max_ut_velocity=float(component.get("max_ut_velocity_mps", 0.0)),
            device=str(self.device),
        )
        channel_model.set_topology(*topology)
        runtime["topology_batch_size"] = batch_size
        ut_loc, bs_loc, ut_orient, bs_orient, ut_velocity, in_state = topology
        del ut_orient, bs_orient
        distance = torch.linalg.vector_norm(ut_loc[:, 0] - bs_loc[:, 0], dim=-1)
        speed = torch.linalg.vector_norm(ut_velocity[:, 0], dim=-1)
        metadata: dict[str, Any] = {
            "scenario": backend,
            "ut_distance_m": distance.view(batch_size, 1).to(torch.float32),
            "ut_speed_mps": speed.view(batch_size, 1).to(torch.float32),
            "indoor_state": in_state[:, 0].view(batch_size, 1),
        }
        los = getattr(channel_model, "los", None)
        if isinstance(los, torch.Tensor) and los.shape[0] == batch_size:
            metadata["los_state"] = los[:, 0].view(batch_size, 1)
        return metadata

    def apply(self, x_rg: torch.Tensor, batch_size: int):
        component_index = self.sampler.next_index()
        component = self.profile["components"][component_index]
        runtime = self._runtimes[component_index]
        metadata: dict[str, Any] = {
            "channel_backend": component["backend"],
            "channel_profile_id": component["id"],
            "channel_profile_name": self.profile["name"],
            "topology_seed": self.topology_seed_base + self.batch_index,
        }
        if component["backend"] == "tdl":
            metadata.update(
                {
                    "scenario": "tdl",
                    "tdl_model": component["tdl_model"],
                    "delay_spread_ns": float(component["delay_spread_ns"]),
                }
            )
        else:
            metadata.update(
                self._prepare_system_topology(
                    component, runtime, batch_size
                )
            )
        y_clean_full, h_full = runtime["channel"](x_rg)
        self.counts[component["id"]] += 1
        self.batch_index += 1
        return y_clean_full, h_full, metadata
