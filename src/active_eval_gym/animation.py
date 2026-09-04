"""Offline CartPole animations derived from saved sweep trajectories."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from active_eval_gym.serialization import read_saved_episode, write_json_new

ANIMATION_SCHEMA_VERSION = 1
RENDERER_ID = "cartpole-overlay-v1"
MANIFEST_FILENAME = "animation-manifest.json"
POLICY_COLORS = ("#0072B2", "#D62728")
PANEL_SIZE = (640, 360)
COMPOSITE_PANEL_SIZE = (480, 270)
MAX_COMPOSITE_CONDITIONS = 6
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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

    @property
    def final_step(self) -> int:
        return len(self.states) - 1


@dataclass(frozen=True)
class ConditionTraces:
    """All policy and seed traces for one sweep condition."""

    condition_id: str
    parameters: dict[str, Any]
    traces: tuple[EpisodeTrace, ...]


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
) -> list[Path]:
    """Create per-condition and/or synchronized CartPole sweep GIFs."""

    if layout not in {"individual", "composite", "both"}:
        raise ValueError("layout must be 'individual', 'composite', or 'both'.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be a positive integer.")

    data = load_cartpole_sweep_animation(evaluation_dir)
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
    color_lookup = dict(zip(data.policy_ids, POLICY_COLORS, strict=False))

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
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "parameters": condition.parameters,
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


def load_cartpole_sweep_animation(evaluation_dir: Path) -> SweepAnimationData:
    """Load and validate the raw trajectories required for animation."""

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
    policy_ids = _required_identifiers(suite.get("policy_ids"), "policy_ids")
    if len(policy_ids) > len(POLICY_COLORS):
        raise ValueError(
            f"CartPole overlay supports at most {len(POLICY_COLORS)} policies."
        )
    seeds = _required_seeds(suite.get("seeds"))
    raw_conditions = suite_record.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("Sweep record must contain a non-empty conditions list.")

    conditions: list[ConditionTraces] = []
    all_traces: list[EpisodeTrace] = []
    for raw_condition in raw_conditions:
        if not isinstance(raw_condition, dict):
            raise ValueError("Every sweep condition must be a mapping.")
        condition_id = _required_identifier(
            raw_condition.get("condition_id"), "condition_id"
        )
        perturbation = raw_condition.get("perturbation")
        parameters = (
            perturbation.get("parameters") if isinstance(perturbation, dict) else None
        )
        if not isinstance(parameters, dict):
            raise ValueError(f"Condition {condition_id!r} has no parameter mapping.")
        traces: list[EpisodeTrace] = []
        for policy_id in policy_ids:
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
                )
                traces.append(trace)
                all_traces.append(trace)
        conditions.append(
            ConditionTraces(
                condition_id=condition_id,
                parameters=dict(parameters),
                traces=tuple(traces),
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
        policy_ids=policy_ids,
        seeds=seeds,
        conditions=tuple(conditions),
        max_episode_steps=max_episode_steps,
        tau=tau,
        x_threshold=x_threshold,
        maximum_pole_length=2 * max(lengths),
    )


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
    geometry = _scene_geometry(data, high_size)
    _draw_background(draw, geometry, data, fonts)

    masks: list[tuple[tuple[int, int, int], Image.Image]] = []
    failures: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    active_counts = {policy_id: 0 for policy_id in data.policy_ids}
    for policy_id in data.policy_ids:
        mask = Image.new("L", high_size, 0)
        run_mask = Image.new("L", high_size, 0)
        for trace in condition.traces:
            if trace.policy_id != policy_id:
                continue
            visible = _visible_state(trace, step, previous_step)
            if visible is None:
                continue
            if visible.active:
                active_counts[policy_id] += 1
            run_mask.paste(0, (0, 0, *high_size))
            run_draw = ImageDraw.Draw(run_mask)
            anchor = _draw_cartpole_mask(
                run_draw,
                visible.state,
                _positive_float(trace.parameters, "length"),
                geometry,
            )
            mask = ImageChops.add(mask, run_mask)
            if visible.failed:
                failures.append((_hex_rgb(colors[policy_id]), anchor))
        masks.append((_hex_rgb(colors[policy_id]), mask))

    base = _composite_density_layers(base, masks)
    draw = ImageDraw.Draw(base)
    _draw_foreground(
        draw,
        condition,
        data,
        step,
        active_counts,
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
    scale: float
    x_limit: float
    cart_width: float
    cart_height: float
    axle_height: float


def _scene_geometry(
    data: SweepAnimationData, size: tuple[int, int]
) -> _SceneGeometry:
    width, height = size
    left, right, top, bottom = 84, 36, 116, 70
    usable_width = width - left - right
    usable_height = height - top - bottom
    x_limit = data.x_threshold + 0.35 * data.maximum_pole_length
    scale = min(
        usable_width / (2 * x_limit),
        usable_height / (1.2 * data.maximum_pole_length),
    )
    return _SceneGeometry(
        width=width,
        height=height,
        center_x=left + usable_width / 2,
        baseline_y=height - bottom,
        scale=scale,
        x_limit=x_limit,
        cart_width=0.42,
        cart_height=0.22,
        axle_height=0.07,
    )


def _draw_background(
    draw: ImageDraw.ImageDraw,
    geometry: _SceneGeometry,
    data: SweepAnimationData,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    interface_scale = 2
    baseline = round(geometry.baseline_y)
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
            (x, 50 * interface_scale, x, baseline + 8 * interface_scale),
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


def _draw_cartpole_mask(
    draw: ImageDraw.ImageDraw,
    state: dict[str, float],
    half_length: float,
    geometry: _SceneGeometry,
) -> tuple[int, int]:
    x = state["cart_position"]
    theta = state["pole_angle"]
    cart_x = geometry.center_x + x * geometry.scale
    cart_center_y = geometry.baseline_y - geometry.cart_height * geometry.scale / 2
    half_width = geometry.cart_width * geometry.scale / 2
    cart_height = geometry.cart_height * geometry.scale
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
        width=max(2, round(geometry.scale * 0.025)),
    )
    axle_y = geometry.baseline_y - geometry.axle_height * geometry.scale
    full_length = 2 * half_length
    end_x = cart_x + full_length * math.sin(theta) * geometry.scale
    end_y = axle_y - full_length * math.cos(theta) * geometry.scale
    draw.line(
        (round(cart_x), round(axle_y), round(end_x), round(end_y)),
        fill=40,
        width=max(4, round(geometry.scale * 0.055)),
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
    return round(cart_x), round(axle_y)


def _draw_foreground(
    draw: ImageDraw.ImageDraw,
    condition: ConditionTraces,
    data: SweepAnimationData,
    step: int,
    active_counts: dict[str, int],
    colors: dict[str, str],
    failures: list[tuple[tuple[int, int, int], tuple[int, int]]],
    geometry: _SceneGeometry,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    interface_scale = 2
    title = _condition_title(condition)
    title_box = draw.textbbox((0, 0), title, font=fonts["title"])
    draw.text(
        (
            (geometry.width - (title_box[2] - title_box[0])) / 2,
            5 * interface_scale,
        ),
        title,
        fill="#222222",
        font=fonts["title"],
    )
    draw.text(
        (12 * interface_scale, 31 * interface_scale),
        f"t = {step * data.tau:.2f} s",
        fill="#333333",
        font=fonts["small"],
    )
    x = max(125 * interface_scale, round(geometry.width * 0.34))
    for policy_id in data.policy_ids:
        label = (
            f"{_short_policy(policy_id)} "
            f"{active_counts[policy_id]}/{len(data.seeds)}"
        )
        draw.rectangle(
            (
                x,
                34 * interface_scale,
                x + 11 * interface_scale,
                45 * interface_scale,
            ),
            fill=colors[policy_id],
        )
        draw.text(
            (x + 16 * interface_scale, 29 * interface_scale),
            label,
            fill="#333333",
            font=fonts["small"],
        )
        label_box = draw.textbbox((0, 0), label, font=fonts["small"])
        x += label_box[2] - label_box[0] + 31 * interface_scale

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
    y = 72 * scale
    for policy_id in data.policy_ids:
        draw.rectangle(
            (30 * scale, y + 5 * scale, 58 * scale, y + 19 * scale),
            fill=colors[policy_id],
        )
        draw.text(
            (70 * scale, y),
            _short_policy(policy_id),
            fill="#222222",
            font=fonts["body"],
        )
        y += 38 * scale
    draw.text(
        (30 * scale, y + 4 * scale),
        f"{len(data.seeds)} paired seeds",
        fill="#444444",
        font=fonts["body"],
    )
    draw.text(
        (30 * scale, y + 38 * scale),
        "Purple indicates policy overlap.",
        fill="#444444",
        font=fonts["small"],
    )
    draw.text(
        (30 * scale, y + 62 * scale),
        "× marks terminal failure; failed runs then disappear.",
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
    if "lqr" in policy_id.lower():
        return "Quantized LQR"
    if "ppo" in policy_id.lower():
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


def _shared_positive_float(traces: list[EpisodeTrace], name: str) -> float:
    values = {_positive_float(trace.parameters, name) for trace in traces}
    if len(values) != 1:
        raise ValueError(f"CartPole parameter {name!r} must be shared by the sweep.")
    return values.pop()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    raw = value.removeprefix("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
