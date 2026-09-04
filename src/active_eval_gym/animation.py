"""Offline CartPole animations derived from saved sweep trajectories."""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from active_eval_gym.serialization import read_saved_episode, write_json_new

ANIMATION_SCHEMA_VERSION = 1
RENDERER_ID = "cartpole-overlay-v1"
MANIFEST_FILENAME = "animation-manifest.json"
POLICY_COLORS = ("#0072B2", "#D62728", "#009E73")
PANEL_SIZE = (640, 360)
COMPOSITE_PANEL_SIZE = (480, 270)
MAX_COMPOSITE_CONDITIONS = 9
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCENE_MARGINS = (84, 36, 116, 70)
_RECOVERY_BAND_FILL = (250, 234, 188)
_RECOVERY_BAND_EDGE = (198, 155, 52)
_RECOVERY_BAND_TEXT = "#8A6714"


@dataclass(frozen=True)
class EpisodeTrace:
    """The render-relevant subset of one hash-verified episode."""

    policy_id: str
    seed: int
    states: tuple[dict[str, float], ...]
    terminated: bool
    truncated: bool
    parameters: dict[str, Any]
    max_episode_steps: int
    trajectory_sha256: str
    squared_angle_prefix: tuple[float, ...] = ()

    @property
    def final_step(self) -> int:
        return len(self.states) - 1


@dataclass(frozen=True)
class ConditionTraces:
    """All policy and seed traces for one sweep condition."""

    condition_id: str
    parameters: dict[str, Any]
    traces: tuple[EpisodeTrace, ...]
    pole_half_length: float = 0.5


@dataclass(frozen=True)
class RecoveryBand:
    """The recovery criterion a boundary study scores its episodes against."""

    rms_angle_degrees: float
    tail_steps: int


@dataclass(frozen=True)
class SweepAnimationData:
    """Validated inputs and shared rendering parameters for a sweep."""

    suite_id: str
    policy_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    conditions: tuple[ConditionTraces, ...]
    max_episode_steps: int
    tau: float
    x_threshold: float
    maximum_pole_length: float
    policy_colors: dict[str, str] = field(default_factory=dict)
    recovery_band: RecoveryBand | None = None


@dataclass(frozen=True)
class VisibleState:
    """One sampled pose and whether it represents terminal failure."""

    state: dict[str, float]
    failed: bool
    active: bool


def animate_cartpole_sweep(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    layout: Literal["individual", "composite", "both"] = "both",
    frame_stride: int = 5,
    condition_ids: Sequence[str] | None = None,
    policy_ids: Sequence[str] | None = None,
) -> list[Path]:
    """Create per-condition and/or synchronized CartPole sweep GIFs."""

    if layout not in {"individual", "composite", "both"}:
        raise ValueError("layout must be 'individual', 'composite', or 'both'.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be a positive integer.")

    data = load_cartpole_sweep_animation(
        evaluation_dir, condition_ids=condition_ids, policy_ids=policy_ids
    )
    if layout in {"composite", "both"} and (
        len(data.conditions) > MAX_COMPOSITE_CONDITIONS
    ):
        raise ValueError(
            "Composite animation supports at most "
            f"{MAX_COMPOSITE_CONDITIONS} conditions; received "
            f"{len(data.conditions)}. Use --layout individual."
        )

    outputs: list[Path] = []
    if layout in {"individual", "both"}:
        outputs.extend(
            output_dir / f"cartpole-overlay-{condition.condition_id}.gif"
            for condition in data.conditions
        )
    if layout in {"composite", "both"}:
        outputs.append(output_dir / "cartpole-overlay-comparison.gif")
    manifest_path = output_dir / MANIFEST_FILENAME
    existing = [path for path in [*outputs, manifest_path] if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite animation artifacts: {names}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = _frame_steps(data.max_episode_steps, frame_stride)
    duration_ms = max(1, round(1000 * data.tau * frame_stride))
    color_lookup = dict(data.policy_colors)

    rendered: list[Path] = []
    if layout in {"individual", "both"}:
        for condition, path in zip(data.conditions, outputs, strict=False):
            _save_gif(
                path,
                (
                    _render_condition_frame(
                        condition,
                        data,
                        step,
                        previous_step,
                        PANEL_SIZE,
                        color_lookup,
                    )
                    for step, previous_step in _step_pairs(steps)
                ),
                duration_ms=duration_ms,
            )
            rendered.append(path)
    if layout in {"composite", "both"}:
        composite_path = output_dir / "cartpole-overlay-comparison.gif"
        _save_gif(
            composite_path,
            (
                _render_composite_frame(
                    data, step, previous_step, color_lookup
                )
                for step, previous_step in _step_pairs(steps)
            ),
            duration_ms=duration_ms,
        )
        rendered.append(composite_path)

    manifest = {
        "schema_version": ANIMATION_SCHEMA_VERSION,
        "renderer_id": RENDERER_ID,
        "suite_id": data.suite_id,
        "environment_id": "CartPole-v1",
        "layout": layout,
        "frame_stride": frame_stride,
        "frame_count": len(steps),
        "simulation_timestep_seconds": data.tau,
        "frame_duration_milliseconds": duration_ms,
        "terminal_behavior": (
            "show the terminal pose once with a failure mark, then remove the run"
        ),
        "policy_colors": color_lookup,
        "selected_policy_ids": list(data.policy_ids),
        "selected_condition_ids": [
            condition.condition_id for condition in data.conditions
        ],
        "recovery_band": (
            None
            if data.recovery_band is None
            else {
                "recovery_rms_angle_deg": data.recovery_band.rms_angle_degrees,
                "recovery_tail_steps": data.recovery_band.tail_steps,
                "source": "suite.boundary_study",
            }
        ),
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "parameters": condition.parameters,
                "pole_half_length": condition.pole_half_length,
                "pole_scale_multipliers": {
                    "individual": _pole_scale_multiplier(data, condition, PANEL_SIZE),
                    "composite": _pole_scale_multiplier(
                        data, condition, COMPOSITE_PANEL_SIZE
                    ),
                },
            }
            for condition in data.conditions
        ],
        "source_trajectories": [
            {
                "condition_id": condition.condition_id,
                "policy_id": trace.policy_id,
                "seed": trace.seed,
                "trajectory_sha256": trace.trajectory_sha256,
            }
            for condition in data.conditions
            for trace in condition.traces
        ],
        "outputs": [
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in rendered
        ],
    }
    write_json_new(manifest_path, manifest)
    return [*rendered, manifest_path]


def load_cartpole_sweep_animation(
    evaluation_dir: Path,
    *,
    condition_ids: Sequence[str] | None = None,
    policy_ids: Sequence[str] | None = None,
) -> SweepAnimationData:
    """Load and validate the raw trajectories required for animation.

    ``condition_ids`` and ``policy_ids`` select subsets of the suite, both kept
    in suite order; an unknown identifier is rejected by name.
    """

    try:
        suite_record = json.loads((evaluation_dir / "suite.json").read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Sweep suite record does not exist: {evaluation_dir / 'suite.json'}."
        ) from error
    suite = suite_record.get("suite")
    if not isinstance(suite, dict):
        raise ValueError("Animation requires a perturbation sweep suite record.")
    if suite.get("environment_id") != "CartPole-v1":
        raise ValueError("CartPole animation requires environment_id='CartPole-v1'.")

    suite_id = _required_identifier(suite.get("suite_id"), "suite_id")
    suite_policy_ids = _required_identifiers(suite.get("policy_ids"), "policy_ids")
    if len(suite_policy_ids) > len(POLICY_COLORS):
        raise ValueError(
            f"CartPole overlay supports at most {len(POLICY_COLORS)} policies."
        )
    suite_colors = dict(zip(suite_policy_ids, POLICY_COLORS, strict=False))
    selected_policy_ids = _select_in_order(
        suite_policy_ids, policy_ids, "policy_ids", "policy"
    )
    recovery_band = _recovery_band(suite.get("boundary_study"))
    seeds = _required_seeds(suite.get("seeds"))
    raw_conditions = suite_record.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("Sweep record must contain a non-empty conditions list.")

    available_condition_ids = [
        _required_identifier(
            raw.get("condition_id") if isinstance(raw, dict) else None, "condition_id"
        )
        for raw in raw_conditions
    ]
    selected_condition_ids = set(
        _select_in_order(
            tuple(available_condition_ids), condition_ids, "condition_ids", "condition"
        )
    )

    conditions: list[ConditionTraces] = []
    all_traces: list[EpisodeTrace] = []
    for raw_condition in raw_conditions:
        if not isinstance(raw_condition, dict):
            raise ValueError("Every sweep condition must be a mapping.")
        condition_id = _required_identifier(
            raw_condition.get("condition_id"), "condition_id"
        )
        if condition_id not in selected_condition_ids:
            continue
        perturbation = raw_condition.get("perturbation")
        parameters = (
            perturbation.get("parameters") if isinstance(perturbation, dict) else None
        )
        if not isinstance(parameters, dict):
            raise ValueError(f"Condition {condition_id!r} has no parameter mapping.")
        traces: list[EpisodeTrace] = []
        for policy_id in selected_policy_ids:
            for seed in seeds:
                episode_dir = (
                    evaluation_dir
                    / "episodes"
                    / policy_id
                    / condition_id
                    / f"seed-{seed:03d}"
                )
                episode = read_saved_episode(episode_dir)
                _validate_episode_identity(
                    episode.metadata,
                    episode_dir,
                    policy_id=policy_id,
                    condition_parameters=parameters,
                    seed=seed,
                )
                resolved = episode.metadata.get("resolved_environment")
                if not isinstance(resolved, dict):
                    raise ValueError(
                        f"Episode {episode_dir} has no resolved environment."
                    )
                if resolved.get("environment_id") != "CartPole-v1":
                    raise ValueError(f"Episode {episode_dir} is not CartPole-v1.")
                env_parameters = resolved.get("parameters")
                if not isinstance(env_parameters, dict):
                    raise ValueError(f"Episode {episode_dir} has no parameter mapping.")
                max_episode_steps = resolved.get("max_episode_steps")
                if (
                    not isinstance(max_episode_steps, int)
                    or isinstance(max_episode_steps, bool)
                    or max_episode_steps <= 0
                ):
                    raise ValueError(
                        f"Episode {episode_dir} has no positive episode limit."
                    )
                states = (episode.reset["environment_state"],) + tuple(
                    transition["environment_state"]
                    for transition in episode.transitions
                )
                _validate_states(states, episode_dir)
                if not episode.transitions:
                    raise ValueError(f"Episode {episode_dir} has no transitions.")
                last = episode.transitions[-1]
                terminated = bool(last.get("terminated", False))
                truncated = bool(last.get("truncated", False))
                if terminated == truncated:
                    raise ValueError(
                        f"Episode {episode_dir} must end by termination or truncation."
                    )
                trace = EpisodeTrace(
                    policy_id=policy_id,
                    seed=seed,
                    states=states,
                    terminated=terminated,
                    truncated=truncated,
                    parameters=dict(env_parameters),
                    max_episode_steps=max_episode_steps,
                    trajectory_sha256=episode.trajectory_sha256,
                    squared_angle_prefix=_squared_angle_prefix(states),
                )
                traces.append(trace)
                all_traces.append(trace)
        conditions.append(
            ConditionTraces(
                condition_id=condition_id,
                parameters=dict(parameters),
                traces=tuple(traces),
                pole_half_length=_shared_positive_float(
                    traces, "length", scope="condition"
                ),
            )
        )

    tau = _shared_positive_float(all_traces, "tau")
    x_threshold = _shared_positive_float(all_traces, "x_threshold")
    lengths = [_positive_float(trace.parameters, "length") for trace in all_traces]
    recorded_max_steps = [trace.final_step for trace in all_traces]
    max_steps_values = {trace.max_episode_steps for trace in all_traces}
    if len(max_steps_values) != 1:
        raise ValueError("CartPole episode limit must be shared by the sweep.")
    resolved_max_steps = max_steps_values.pop()
    nominal = suite_record.get("nominal_environment")
    nominal_max_steps = (
        nominal.get("max_episode_steps") if isinstance(nominal, dict) else None
    )
    if isinstance(nominal_max_steps, int) and nominal_max_steps > 0:
        if nominal_max_steps != resolved_max_steps:
            raise ValueError(
                "Resolved CartPole episode limit does not match the nominal suite."
            )
        max_episode_steps = nominal_max_steps
    else:
        max_episode_steps = resolved_max_steps
    if max(recorded_max_steps) > max_episode_steps:
        raise ValueError("A saved trajectory exceeds the declared episode limit.")

    return SweepAnimationData(
        suite_id=suite_id,
        policy_ids=selected_policy_ids,
        seeds=seeds,
        conditions=tuple(conditions),
        max_episode_steps=max_episode_steps,
        tau=tau,
        x_threshold=x_threshold,
        maximum_pole_length=2 * max(lengths),
        policy_colors={
            policy_id: suite_colors[policy_id] for policy_id in selected_policy_ids
        },
        recovery_band=recovery_band,
    )


def _select_in_order(
    available: tuple[str, ...],
    requested: Sequence[str] | None,
    field_name: str,
    noun: str,
) -> tuple[str, ...]:
    """Restrict ``available`` to ``requested`` while preserving suite order."""

    if requested is None:
        return available
    wanted = list(dict.fromkeys(requested))
    if not wanted:
        raise ValueError(f"{field_name} selection must not be empty.")
    unknown = [value for value in wanted if value not in available]
    if unknown:
        raise ValueError(
            f"Unknown {noun} identifier(s) {', '.join(sorted(unknown))!r}; "
            f"the suite declares {_summarize_identifiers(available)}."
        )
    return tuple(value for value in available if value in set(wanted))


def _summarize_identifiers(available: tuple[str, ...], limit: int = 8) -> str:
    """Name a few valid identifiers without dumping a 248-condition suite."""

    shown = ", ".join(available[:limit])
    if len(available) <= limit:
        return shown
    return f"{shown}, ... ({len(available)} in total)"


def _recovery_band(raw: Any) -> RecoveryBand | None:
    """Read the recovery criterion a boundary study declares, if any."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("boundary_study must be a mapping when present.")
    angle = raw.get("recovery_rms_angle_deg")
    tail = raw.get("recovery_tail_steps")
    if (
        not isinstance(angle, (int, float))
        or isinstance(angle, bool)
        or not math.isfinite(float(angle))
        or angle <= 0
    ):
        raise ValueError("boundary_study.recovery_rms_angle_deg must be positive.")
    if not isinstance(tail, int) or isinstance(tail, bool) or tail <= 0:
        raise ValueError("boundary_study.recovery_tail_steps must be a positive int.")
    return RecoveryBand(rms_angle_degrees=float(angle), tail_steps=tail)


def _squared_angle_prefix(states: tuple[dict[str, float], ...]) -> tuple[float, ...]:
    """Prefix sums of squared pole angle, for O(1) trailing-window RMS."""

    prefix = [0.0]
    total = 0.0
    for state in states:
        total += float(state["pole_angle"]) ** 2
        prefix.append(total)
    return tuple(prefix)


def _trailing_rms_degrees(trace: EpisodeTrace, step: int, tail_steps: int) -> float:
    """Root-mean-square |pole angle| over the trailing window ending at ``step``."""

    end = min(step, trace.final_step)
    start = max(0, end - tail_steps + 1)
    count = end - start + 1
    if trace.squared_angle_prefix:
        total = trace.squared_angle_prefix[end + 1] - trace.squared_angle_prefix[start]
    else:
        total = sum(
            float(state["pole_angle"]) ** 2 for state in trace.states[start : end + 1]
        )
    return math.degrees(math.sqrt(max(total, 0.0) / count))


def _render_composite_frame(
    data: SweepAnimationData,
    step: int,
    previous_step: int,
    colors: dict[str, str],
) -> Image.Image:
    slot_count = len(data.conditions) + 1
    columns = min(3, math.ceil(math.sqrt(slot_count)))
    rows = math.ceil(slot_count / columns)
    panel_width, panel_height = COMPOSITE_PANEL_SIZE
    canvas = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    for index, condition in enumerate(data.conditions):
        panel = _render_condition_frame(
            condition,
            data,
            step,
            previous_step,
            COMPOSITE_PANEL_SIZE,
            colors,
        )
        canvas.paste(
            panel,
            (
                (index % columns) * panel_width,
                (index // columns) * panel_height,
            ),
        )
    legend_index = len(data.conditions)
    legend = _render_legend_panel(data, colors, COMPOSITE_PANEL_SIZE)
    canvas.paste(
        legend,
        (
            (legend_index % columns) * panel_width,
            (legend_index // columns) * panel_height,
        ),
    )
    return canvas


def _render_condition_frame(
    condition: ConditionTraces,
    data: SweepAnimationData,
    step: int,
    previous_step: int,
    size: tuple[int, int],
    colors: dict[str, str],
) -> Image.Image:
    scale_factor = 2
    width, height = size
    high_size = (width * scale_factor, height * scale_factor)
    base = Image.new("RGB", high_size, "white")
    draw = ImageDraw.Draw(base)
    fonts = _fonts(scale_factor, compact=width <= COMPOSITE_PANEL_SIZE[0])
    geometry = _scene_geometry(
        data, high_size, pole_half_length=condition.pole_half_length
    )
    _draw_background(draw, geometry, data, fonts)

    tail_steps = data.recovery_band.tail_steps if data.recovery_band else None
    masks: list[tuple[tuple[int, int, int], Image.Image]] = []
    failures: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    active_counts = {policy_id: 0 for policy_id in data.policy_ids}
    trailing_rms: dict[str, float | None] = {}
    for policy_id in data.policy_ids:
        mask = Image.new("L", high_size, 0)
        run_mask = Image.new("L", high_size, 0)
        active_rms: list[float] = []
        for trace in condition.traces:
            if trace.policy_id != policy_id:
                continue
            visible = _visible_state(trace, step, previous_step)
            if visible is None:
                continue
            if visible.active:
                active_counts[policy_id] += 1
                if tail_steps is not None:
                    active_rms.append(
                        _trailing_rms_degrees(trace, step, tail_steps)
                    )
            run_draw = ImageDraw.Draw(run_mask)
            anchor, box = _draw_cartpole_mask(
                run_draw,
                visible.state,
                geometry,
            )
            if box is not None:
                mask.paste(
                    ImageChops.add(mask.crop(box), run_mask.crop(box)), box
                )
                run_mask.paste(0, box)
            if visible.failed:
                failures.append((_hex_rgb(colors[policy_id]), anchor))
        masks.append((_hex_rgb(colors[policy_id]), mask))
        trailing_rms[policy_id] = (
            sum(active_rms) / len(active_rms) if active_rms else None
        )

    base = _composite_density_layers(base, masks)
    draw = ImageDraw.Draw(base)
    _draw_foreground(
        draw,
        condition,
        data,
        step,
        active_counts,
        trailing_rms,
        colors,
        failures,
        geometry,
        fonts,
    )
    return base.resize(size, Image.Resampling.LANCZOS)


def _composite_density_layers(
    base: Image.Image,
    layers: list[tuple[tuple[int, int, int], Image.Image]],
) -> Image.Image:
    """Blend colored density masks symmetrically, independent of draw order."""

    base_values = np.asarray(base, dtype=np.float32)
    strengths = [np.asarray(mask, dtype=np.float32) / 255 for _, mask in layers]
    if not strengths:
        return base.copy()
    total_strength = np.zeros_like(strengths[0])
    for strength in strengths:
        total_strength += strength
    opacity = 1 - np.prod(
        np.stack([1 - strength for strength in strengths], axis=0), axis=0
    )
    color_sum = np.zeros_like(base_values)
    for (color, _), strength in zip(layers, strengths, strict=True):
        color_sum += strength[..., None] * np.asarray(color, dtype=np.float32)
    mixture = np.divide(
        color_sum,
        total_strength[..., None],
        out=np.zeros_like(color_sum),
        where=total_strength[..., None] > 0,
    )
    result = base_values * (1 - opacity[..., None]) + mixture * opacity[..., None]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _visible_state(
    trace: EpisodeTrace, step: int, previous_step: int
) -> VisibleState | None:
    if step <= trace.final_step:
        failed = trace.terminated and step == trace.final_step
        return VisibleState(trace.states[step], failed=failed, active=not failed)
    if trace.terminated and previous_step < trace.final_step < step:
        return VisibleState(trace.states[-1], failed=True, active=False)
    return None


@dataclass(frozen=True)
class _SceneGeometry:
    width: int
    height: int
    center_x: float
    baseline_y: float
    scene_top: float
    scale: float
    pole_scale: float
    pole_full_length: float
    x_limit: float
    cart_width: float
    cart_height: float
    axle_height: float

    @property
    def pole_pixel_length(self) -> float:
        return self.pole_full_length * self.pole_scale

    @property
    def pole_scale_multiplier(self) -> float:
        return self.pole_scale / self.scale


def _scene_geometry(
    data: SweepAnimationData,
    size: tuple[int, int],
    *,
    pole_half_length: float,
) -> _SceneGeometry:
    """Shared cart-position scale, per-condition pole scale.

    Cart positions and the +/- x_threshold markers use one scale for every
    condition so panels stay comparable, while the pole is scaled to its own
    length so that a 0.02 m pole is as visible as a 3.0 m one.
    """

    width, height = size
    left, right, top, bottom = _SCENE_MARGINS
    usable_width = width - left - right
    usable_height = height - top - bottom
    x_limit = data.x_threshold + 0.35 * data.maximum_pole_length
    scale = min(
        usable_width / (2 * x_limit),
        usable_height / (1.2 * data.maximum_pole_length),
    )
    pole_full_length = 2 * pole_half_length
    pole_scale = usable_height / (1.2 * pole_full_length)
    return _SceneGeometry(
        width=width,
        height=height,
        center_x=left + usable_width / 2,
        baseline_y=height - bottom,
        scene_top=float(top),
        scale=scale,
        pole_scale=pole_scale,
        pole_full_length=pole_full_length,
        x_limit=x_limit,
        cart_width=0.42,
        cart_height=0.22,
        axle_height=0.07,
    )


def _pole_scale_multiplier(
    data: SweepAnimationData,
    condition: ConditionTraces,
    size: tuple[int, int],
) -> float:
    """How much the pole is magnified relative to the cart-position scale."""

    high_size = (size[0] * 2, size[1] * 2)
    geometry = _scene_geometry(
        data, high_size, pole_half_length=condition.pole_half_length
    )
    return geometry.pole_scale_multiplier


def _draw_background(
    draw: ImageDraw.ImageDraw,
    geometry: _SceneGeometry,
    data: SweepAnimationData,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    interface_scale = 2
    baseline = round(geometry.baseline_y)
    if data.recovery_band is not None:
        _draw_recovery_wedge(draw, geometry, data.recovery_band, fonts)
    draw.line(
        (
            20 * interface_scale,
            baseline,
            geometry.width - 15 * interface_scale,
            baseline,
        ),
        fill="#777777",
        width=2 * interface_scale,
    )
    for threshold in (-data.x_threshold, data.x_threshold):
        x = round(geometry.center_x + threshold * geometry.scale)
        draw.line(
            (x, round(geometry.scene_top), x, baseline + 8 * interface_scale),
            fill="#C5C5C5",
            width=2 * interface_scale,
        )
        label = f"{threshold:g}"
        box = draw.textbbox((0, 0), label, font=fonts["small"])
        draw.text(
            (
                x - (box[2] - box[0]) / 2,
                baseline + 10 * interface_scale,
            ),
            label,
            fill="#666666",
            font=fonts["small"],
        )
    zero_x = round(geometry.center_x)
    draw.line(
        (zero_x, baseline, zero_x, baseline + 6 * interface_scale),
        fill="#777777",
        width=2 * interface_scale,
    )


def _draw_recovery_wedge(
    draw: ImageDraw.ImageDraw,
    geometry: _SceneGeometry,
    band: RecoveryBand,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    """Shade the +/- recovery-RMS wedge the study scores settling against."""

    interface_scale = 2
    axle_y = geometry.baseline_y - geometry.axle_height * geometry.scale
    reach = geometry.pole_pixel_length
    angle = math.radians(band.rms_angle_degrees)
    left_x = geometry.center_x - reach * math.sin(angle)
    right_x = geometry.center_x + reach * math.sin(angle)
    tip_y = axle_y - reach * math.cos(angle)
    draw.polygon(
        [
            (geometry.center_x, axle_y),
            (left_x, tip_y),
            (right_x, tip_y),
        ],
        fill=_RECOVERY_BAND_FILL,
    )
    for x_tip in (left_x, right_x):
        draw.line(
            (geometry.center_x, axle_y, x_tip, tip_y),
            fill=_RECOVERY_BAND_EDGE,
            width=interface_scale,
        )
    label = f"recovery band ±{band.rms_angle_degrees:g}°"
    box = draw.textbbox((0, 0), label, font=fonts["small"])
    draw.text(
        (
            geometry.center_x - (box[2] - box[0]) / 2,
            max(geometry.scene_top, tip_y - (box[3] - box[1]) - 6 * interface_scale),
        ),
        label,
        fill=_RECOVERY_BAND_TEXT,
        font=fonts["small"],
    )


def _draw_cartpole_mask(
    draw: ImageDraw.ImageDraw,
    state: dict[str, float],
    geometry: _SceneGeometry,
) -> tuple[tuple[int, int], tuple[int, int, int, int] | None]:
    """Draw one run and report its anchor plus the touched bounding box."""

    x = state["cart_position"]
    theta = state["pole_angle"]
    cart_x = geometry.center_x + x * geometry.scale
    cart_center_y = geometry.baseline_y - geometry.cart_height * geometry.scale / 2
    half_width = geometry.cart_width * geometry.scale / 2
    cart_height = geometry.cart_height * geometry.scale
    outline_width = max(2, round(geometry.scale * 0.025))
    draw.rounded_rectangle(
        (
            round(cart_x - half_width),
            round(cart_center_y - cart_height / 2),
            round(cart_x + half_width),
            round(cart_center_y + cart_height / 2),
        ),
        radius=max(2, round(cart_height * 0.12)),
        fill=34,
        outline=52,
        width=outline_width,
    )
    axle_y = geometry.baseline_y - geometry.axle_height * geometry.scale
    pole_pixels = geometry.pole_pixel_length
    end_x = cart_x + pole_pixels * math.sin(theta)
    end_y = axle_y - pole_pixels * math.cos(theta)
    pole_width = max(4, round(geometry.scale * 0.055))
    draw.line(
        (round(cart_x), round(axle_y), round(end_x), round(end_y)),
        fill=40,
        width=pole_width,
    )
    radius = max(3, round(geometry.scale * 0.045))
    draw.ellipse(
        (
            round(cart_x - radius),
            round(axle_y - radius),
            round(cart_x + radius),
            round(axle_y + radius),
        ),
        fill=48,
    )
    pad = max(outline_width, pole_width, radius) + 2
    box = _clipped_box(
        min(cart_x - half_width, end_x, cart_x - radius) - pad,
        min(cart_center_y - cart_height / 2, end_y, axle_y - radius) - pad,
        max(cart_x + half_width, end_x, cart_x + radius) + pad,
        max(cart_center_y + cart_height / 2, end_y, axle_y + radius) + pad,
        geometry.width,
        geometry.height,
    )
    return (round(cart_x), round(axle_y)), box


def _clipped_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    box = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(width, math.ceil(right)),
        min(height, math.ceil(bottom)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _draw_foreground(
    draw: ImageDraw.ImageDraw,
    condition: ConditionTraces,
    data: SweepAnimationData,
    step: int,
    active_counts: dict[str, int],
    trailing_rms: dict[str, float | None],
    colors: dict[str, str],
    failures: list[tuple[tuple[int, int, int], tuple[int, int]]],
    geometry: _SceneGeometry,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    interface_scale = 2
    title = _condition_title(condition)
    title_font = _fitted_font(draw, title, fonts, geometry.width - 16)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        (
            (geometry.width - (title_box[2] - title_box[0])) / 2,
            5 * interface_scale,
        ),
        title,
        fill="#222222",
        font=title_font,
    )
    clock = f"t = {step * data.tau:.2f} s"
    draw.text(
        (12 * interface_scale, 31 * interface_scale),
        clock,
        fill="#333333",
        font=fonts["small"],
    )
    clock_box = draw.textbbox((0, 0), clock, font=fonts["small"])
    clock_end = 12 * interface_scale + (clock_box[2] - clock_box[0])
    _draw_policy_readout(
        draw,
        data,
        active_counts,
        trailing_rms,
        colors,
        geometry,
        fonts,
        left_limit=clock_end + 10 * interface_scale,
    )
    _draw_pole_scale_note(draw, geometry, condition, fonts)
    if data.recovery_band is not None:
        _draw_recovery_timeline(draw, geometry, data, step, fonts)

    failure_width = max(3, round(geometry.scale * 0.025))
    failure_radius = max(7, round(geometry.scale * 0.08))
    for color, (x_position, y_position) in failures:
        draw.line(
            (
                x_position - failure_radius,
                y_position - failure_radius,
                x_position + failure_radius,
                y_position + failure_radius,
            ),
            fill=color,
            width=failure_width,
        )
        draw.line(
            (
                x_position - failure_radius,
                y_position + failure_radius,
                x_position + failure_radius,
                y_position - failure_radius,
            ),
            fill=color,
            width=failure_width,
        )


def _draw_policy_readout(
    draw: ImageDraw.ImageDraw,
    data: SweepAnimationData,
    active_counts: dict[str, int],
    trailing_rms: dict[str, float | None],
    colors: dict[str, str],
    geometry: _SceneGeometry,
    fonts: dict[str, ImageFont.ImageFont],
    *,
    left_limit: float,
) -> None:
    """Right-align the surviving-seed and trailing-RMS chips into the header."""

    interface_scale = 2
    swatch = 11 * interface_scale
    padding = 5 * interface_scale
    labels = [
        _policy_readout_label(
            policy_id,
            active_counts[policy_id],
            len(data.seeds),
            trailing_rms.get(policy_id),
            data.recovery_band,
        )
        for policy_id in data.policy_ids
    ]
    widths = [
        swatch + padding + (draw.textbbox((0, 0), label, font=fonts["small"])[2])
        for label in labels
    ]
    right_edge = geometry.width - 12 * interface_scale
    for gap in (24 * interface_scale, 14 * interface_scale, 8 * interface_scale):
        total = sum(widths) + gap * max(0, len(widths) - 1)
        x = max(left_limit, right_edge - total)
        if x + total <= right_edge:
            break
    for policy_id, label, width in zip(data.policy_ids, labels, widths, strict=True):
        draw.rectangle(
            (
                x,
                34 * interface_scale,
                x + swatch,
                45 * interface_scale,
            ),
            fill=colors[policy_id],
        )
        draw.text(
            (x + swatch + padding, 29 * interface_scale),
            label,
            fill="#333333",
            font=fonts["small"],
        )
        x += width + gap


def _policy_readout_label(
    policy_id: str,
    active: int,
    seed_count: int,
    trailing_rms: float | None,
    band: RecoveryBand | None,
) -> str:
    label = f"{_short_policy(policy_id)} {active}/{seed_count}"
    if band is None:
        return label
    if trailing_rms is None:
        return f"{label} --"
    precision = 1 if trailing_rms < 10 else 0
    return f"{label} {trailing_rms:.{precision}f}°"


def _fitted_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    fonts: dict[str, ImageFont.ImageFont],
    available_width: float,
) -> ImageFont.ImageFont:
    """Pick the largest configured font whose rendering fits the panel."""

    for name in ("title", "body", "small"):
        font = fonts[name]
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= available_width:
            return font
    return fonts["small"]


def _draw_pole_scale_note(
    draw: ImageDraw.ImageDraw,
    geometry: _SceneGeometry,
    condition: ConditionTraces,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    """State this panel's own pole scale, which differs between conditions."""

    interface_scale = 2
    note = (
        f"pole {condition.pole_half_length:g} m drawn ×"
        f"{geometry.pole_scale_multiplier:.2g}"
    )
    draw.text(
        (12 * interface_scale, 53 * interface_scale),
        note,
        fill="#666666",
        font=fonts["small"],
    )


def _draw_recovery_timeline(
    draw: ImageDraw.ImageDraw,
    geometry: _SceneGeometry,
    data: SweepAnimationData,
    step: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    """Show where the trailing recovery-scoring window sits in the episode."""

    band = data.recovery_band
    if band is None:
        return
    interface_scale = 2
    left, right = _SCENE_MARGINS[0], geometry.width - _SCENE_MARGINS[1]
    top = 46 * interface_scale
    bottom = 52 * interface_scale
    span = right - left

    def position(value: int) -> float:
        return left + span * min(1.0, max(0.0, value / data.max_episode_steps))

    draw.rectangle((left, top, right, bottom), fill="#E6E6E6")
    window_start = position(max(0, data.max_episode_steps - band.tail_steps))
    draw.rectangle(
        (window_start, top, right, bottom),
        fill=_RECOVERY_BAND_FILL,
        outline=_RECOVERY_BAND_EDGE,
    )
    playhead = position(step)
    middle = (top + bottom) / 2
    draw.line(
        (left, middle, playhead, middle),
        fill="#8A8A8A",
        width=2 * interface_scale,
    )
    draw.line(
        (playhead, top - 2 * interface_scale, playhead, bottom + 2 * interface_scale),
        fill="#333333",
        width=interface_scale,
    )
    label = f"recovery window: last {band.tail_steps} steps"
    box = draw.textbbox((0, 0), label, font=fonts["small"])
    label_x = min(right - (box[2] - box[0]), max(left, window_start))
    draw.text(
        (label_x, bottom + 2 * interface_scale),
        label,
        fill=_RECOVERY_BAND_TEXT,
        font=fonts["small"],
    )


def _render_legend_panel(
    data: SweepAnimationData,
    colors: dict[str, str],
    size: tuple[int, int],
) -> Image.Image:
    scale = 2
    image = Image.new("RGB", (size[0] * scale, size[1] * scale), "#F7F7F7")
    draw = ImageDraw.Draw(image)
    fonts = _fonts(scale, compact=True)
    draw.text(
        (28 * scale, 24 * scale),
        data.suite_id,
        fill="#222222",
        font=fonts["title"],
    )
    y = 54 * scale
    for policy_id in data.policy_ids:
        draw.rectangle(
            (30 * scale, y + 4 * scale, 58 * scale, y + 17 * scale),
            fill=colors[policy_id],
        )
        draw.text(
            (70 * scale, y),
            _short_policy(policy_id),
            fill="#222222",
            font=fonts["body"],
        )
        y += 28 * scale
    draw.text(
        (30 * scale, y + 2 * scale),
        f"{len(data.seeds)} paired seeds",
        fill="#444444",
        font=fonts["body"],
    )
    notes = [
        "Blended colour indicates policy overlap.",
        "× marks terminal failure; failed runs then disappear.",
        "Each panel scales its own pole to its own length;",
        "cart positions and ±x_threshold share one scale.",
    ]
    if data.recovery_band is not None:
        notes.extend(
            [
                (
                    f"Amber wedge = ±{data.recovery_band.rms_angle_degrees:g}° "
                    "recovery band; amber bar = scoring window."
                ),
                (
                    "Readout: live seeds, then their mean trailing-"
                    f"{data.recovery_band.tail_steps}-step RMS |θ|."
                ),
                (
                    f"Recovery = survive {data.max_episode_steps} steps AND "
                    "settle inside the band."
                ),
                "Survival alone is not recovery.",
            ]
        )
    for index, note in enumerate(notes):
        draw.text(
            (30 * scale, y + (22 + 13 * index) * scale),
            note,
            fill="#444444",
            font=fonts["small"],
        )
    return image.resize(size, Image.Resampling.LANCZOS)


def _fonts(
    scale: int, *, compact: bool
) -> dict[str, ImageFont.ImageFont]:
    sizes = (16, 12, 10) if compact else (19, 14, 11)
    try:
        return {
            "title": ImageFont.truetype("DejaVuSans.ttf", sizes[0] * scale),
            "body": ImageFont.truetype("DejaVuSans.ttf", sizes[1] * scale),
            "small": ImageFont.truetype("DejaVuSans.ttf", sizes[2] * scale),
        }
    except OSError:
        fallback = ImageFont.load_default()
        return {"title": fallback, "body": fallback, "small": fallback}


def _save_gif(path: Path, frames: Any, *, duration_ms: int) -> None:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("Cannot save an animation with no frames.") from error
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        first.save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=iterator,
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=True,
        )
        temporary.rename(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _frame_steps(max_episode_steps: int, stride: int) -> tuple[int, ...]:
    values = list(range(0, max_episode_steps + 1, stride))
    if values[-1] != max_episode_steps:
        values.append(max_episode_steps)
    return tuple(values)


def _step_pairs(steps: tuple[int, ...]):
    previous = -1
    for step in steps:
        yield step, previous
        previous = step


def _condition_title(condition: ConditionTraces) -> str:
    labels = {
        "dropout_probability": "Dropout p",
        "delay_steps": "Action delay",
        "pole_angle_noise_std_deg": "Pole-angle noise SD (deg)",
        "force_mag": "Force (N)",
        "masspole": "Pole mass",
        "delta_theta_deg": "Initial angle (deg)",
        "initial_theta_deg": "Initial angle (deg)",
        "theta_threshold_deg": "Angle limit (deg)",
        "length": "Pole half-length",
    }
    parts = [
        f"{labels.get(name, name)} = {value:g}"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else f"{labels.get(name, name)} = {value}"
        for name, value in condition.parameters.items()
    ]
    return ", ".join(parts) or condition.condition_id


def _short_policy(policy_id: str) -> str:
    lowered = policy_id.lower()
    if "lqr" in lowered:
        return "Quantized LQR"
    if "ppo" in lowered and "antisymmetrized" in lowered:
        return "PPO anti-sym"
    if "ppo" in lowered:
        return "PPO"
    return policy_id


def _required_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a filesystem-safe non-empty identifier.")
    return value


def _required_identifiers(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    values = tuple(_required_identifier(item, name) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be unique.")
    return values


def _required_seeds(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError("seeds must be a non-empty list of integers.")
    values = tuple(value)
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique.")
    return values


def _validate_episode_identity(
    metadata: dict[str, Any],
    episode_dir: Path,
    *,
    policy_id: str,
    condition_parameters: dict[str, Any],
    seed: int,
) -> None:
    if metadata.get("episode_seed") != seed:
        raise ValueError(f"Episode {episode_dir} does not record seed {seed}.")
    artifact = metadata.get("policy_artifact")
    if not isinstance(artifact, dict) or artifact.get("policy_id") != policy_id:
        raise ValueError(f"Episode {episode_dir} does not record policy {policy_id!r}.")
    perturbation = metadata.get("perturbation")
    recorded_parameters = (
        perturbation.get("parameters") if isinstance(perturbation, dict) else None
    )
    if recorded_parameters != condition_parameters:
        raise ValueError(
            f"Episode {episode_dir} does not match its declared sweep condition."
        )


def _validate_states(states: tuple[dict[str, float], ...], episode_dir: Path) -> None:
    required = {
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_angular_velocity",
    }
    for index, state in enumerate(states):
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError(
                f"Episode {episode_dir} state {index} is not a CartPole state."
            )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in state.values()
        ):
            raise ValueError(f"Episode {episode_dir} state {index} is non-finite.")


def _positive_float(parameters: dict[str, Any], name: str) -> float:
    value = parameters.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"Resolved CartPole parameter {name!r} must be positive.")
    return float(value)


def _shared_positive_float(
    traces: list[EpisodeTrace], name: str, *, scope: str = "sweep"
) -> float:
    values = {_positive_float(trace.parameters, name) for trace in traces}
    if len(values) != 1:
        raise ValueError(f"CartPole parameter {name!r} must be shared by the {scope}.")
    return values.pop()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    raw = value.removeprefix("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
