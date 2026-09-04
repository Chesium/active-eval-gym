"""Command-line workflows for policy construction and fixed evaluation."""

import argparse
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from active_eval_gym.animation import animate_cartpole_sweep
from active_eval_gym.boundary import plan_boundary_stage
from active_eval_gym.config import (
    load_cartpole_symmetry_suite,
    load_nominal_env_spec,
    load_nominal_suite,
    load_sweep_suite,
)
from active_eval_gym.envs.factory import SUPPORTED_ENVIRONMENTS, make_environment
from active_eval_gym.envs.specs import (
    IDENTITY_OBSERVATION,
    apply_observation_adapter,
    capture_resolved_environment,
    package_versions,
)
from active_eval_gym.evaluate import (
    evaluate_nominal_suite,
    freeze_candidate,
    make_artifact_environment,
)
from active_eval_gym.metrics import compute_metrics
from active_eval_gym.models import PolicyArtifactMetadata, PolicyDesignSpec
from active_eval_gym.plotting import plot_sweep
from active_eval_gym.policies.artifacts import (
    build_lqr_artifact,
    load_policy_artifact,
    train_learned_artifact,
)
from active_eval_gym.policies.base import ConstantPolicy, zero_action
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    ensure_output_available,
    write_episode,
    write_metrics,
)
from active_eval_gym.sweeps import analyze_sweep, collect_sweep
from active_eval_gym.symmetry import run_cartpole_symmetry_study

CONSTANT_POLICY_ID = "constant-zero-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_CONFIGS = {
    "CartPole-v1": "cartpole_v1_nominal.toml",
    "Pendulum-v1": "pendulum_v1_nominal.toml",
    "MiniGrid-Empty-8x8-v0": "minigrid_empty8x8_v0_nominal.toml",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without executing a workflow."""

    parser = argparse.ArgumentParser(prog="active-eval-gym")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout_parser = subparsers.add_parser(
        "rollout", help="collect and save one fixed-policy episode"
    )
    rollout_parser.add_argument(
        "--env", required=True, choices=SUPPORTED_ENVIRONMENTS, dest="env_id"
    )
    rollout_parser.add_argument("--seed", required=True, type=int)
    rollout_parser.add_argument("--output", required=True, type=Path)
    rollout_parser.add_argument("--policy-artifact", type=Path)
    rollout_parser.set_defaults(handler=_run_rollout)

    build_policy_parser = subparsers.add_parser(
        "build-policy", help="build a classical policy candidate"
    )
    _add_config_output_arguments(build_policy_parser)
    build_policy_parser.set_defaults(handler=_run_build_policy)

    train_policy_parser = subparsers.add_parser(
        "train-policy", help="train a learned policy candidate"
    )
    _add_config_output_arguments(train_policy_parser)
    train_policy_parser.set_defaults(handler=_run_train_policy)

    freeze_parser = subparsers.add_parser(
        "freeze-policy", help="validate and freeze one candidate policy"
    )
    freeze_parser.add_argument("--artifact", required=True, type=Path)
    freeze_parser.add_argument("--suite", required=True, type=Path)
    freeze_parser.set_defaults(handler=_run_freeze_policy)

    evaluate_parser = subparsers.add_parser(
        "evaluate-nominal", help="run the declared nominal multi-seed suite"
    )
    evaluate_parser.add_argument("--suite", required=True, type=Path)
    evaluate_parser.add_argument("--artifact-root", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser.set_defaults(handler=_run_evaluate_nominal)

    render_parser = subparsers.add_parser(
        "render-policy", help="inspect one frozen policy in human rendering mode"
    )
    render_parser.add_argument("--artifact", required=True, type=Path)
    render_parser.add_argument("--seed", required=True, type=int)
    render_parser.set_defaults(handler=_run_render_policy)

    collect_sweep_parser = subparsers.add_parser(
        "collect-sweep", help="collect a frozen-policy perturbation grid"
    )
    collect_sweep_parser.add_argument("--suite", required=True, type=Path)
    collect_sweep_parser.add_argument("--artifact-root", required=True, type=Path)
    collect_sweep_parser.add_argument("--output", required=True, type=Path)
    collect_sweep_parser.set_defaults(handler=_run_collect_sweep)

    analyze_sweep_parser = subparsers.add_parser(
        "analyze-sweep", help="derive versioned metrics from raw sweep trajectories"
    )
    analyze_sweep_parser.add_argument("--evaluation", required=True, type=Path)
    analyze_sweep_parser.set_defaults(handler=_run_analyze_sweep)

    plot_sweep_parser = subparsers.add_parser(
        "plot-sweep", help="plot a completed perturbation analysis"
    )
    plot_sweep_parser.add_argument("--evaluation", required=True, type=Path)
    plot_sweep_parser.add_argument("--output", required=True, type=Path)
    plot_sweep_parser.set_defaults(handler=_run_plot_sweep)

    animate_sweep_parser = subparsers.add_parser(
        "animate-sweep",
        help="animate saved CartPole sweep trajectories",
    )
    animate_sweep_parser.add_argument("--evaluation", required=True, type=Path)
    animate_sweep_parser.add_argument("--output", required=True, type=Path)
    animate_sweep_parser.add_argument(
        "--layout",
        choices=("individual", "composite", "both"),
        default="both",
    )
    animate_sweep_parser.add_argument("--frame-stride", type=int, default=5)
    animate_sweep_parser.add_argument(
        "--condition",
        action="append",
        default=None,
        metavar="CONDITION_ID",
        help=(
            "restrict the animation to these sweep conditions; repeatable and "
            "comma-separated, kept in suite order"
        ),
    )
    animate_sweep_parser.add_argument(
        "--policy",
        action="append",
        default=None,
        metavar="POLICY_ID",
        help=(
            "restrict the animation to these policies; repeatable and "
            "comma-separated, kept in suite order"
        ),
    )
    animate_sweep_parser.set_defaults(handler=_run_animate_sweep)

    boundary_parser = subparsers.add_parser(
        "select-boundary-sweep",
        help="select an immutable adaptive CartPole boundary stage",
    )
    boundary_parser.add_argument("--pilot", required=True, type=Path)
    boundary_parser.add_argument(
        "--evaluation", required=True, action="append", type=Path
    )
    boundary_parser.add_argument(
        "--stage", required=True, choices=("refinement-1", "refinement-2", "final")
    )
    boundary_parser.add_argument("--output", required=True, type=Path)
    boundary_parser.set_defaults(handler=_run_select_boundary_sweep)

    symmetry_parser = subparsers.add_parser(
        "evaluate-cartpole-symmetry",
        help="audit a PPO and run exact CartPole mirror interventions",
    )
    symmetry_parser.add_argument("--suite", required=True, type=Path)
    symmetry_parser.add_argument("--artifact-root", required=True, type=Path)
    symmetry_parser.add_argument("--source-evaluation", required=True, type=Path)
    symmetry_parser.add_argument("--output", required=True, type=Path)
    symmetry_parser.add_argument("--figure", required=True, type=Path)
    symmetry_parser.set_defaults(handler=_run_cartpole_symmetry)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))


def _run_rollout(args: argparse.Namespace) -> int:
    ensure_output_available(args.output)
    if args.policy_artifact is None:
        nominal = _load_default_nominal(args.env_id)
        env = make_environment(args.env_id)
        env = apply_observation_adapter(env, IDENTITY_OBSERVATION)
        resolved = capture_resolved_environment(env, nominal, IDENTITY_OBSERVATION)
        policy = ConstantPolicy(zero_action(env.action_space))
        metadata = _builtin_artifact(nominal)
    else:
        policy, metadata = load_policy_artifact(args.policy_artifact)
        if metadata.design_spec.environment.environment_id != args.env_id:
            raise ValueError(
                f"Policy {metadata.policy_id!r} requires "
                f"{metadata.design_spec.environment.environment_id}, not {args.env_id}."
            )
        env, resolved = make_artifact_environment(metadata)
        nominal = metadata.design_spec.environment
    try:
        episode = collect_episode(
            env,
            policy,
            policy_artifact=metadata,
            nominal_environment=nominal,
            resolved_environment=resolved,
            episode_seed=args.seed,
        )
    finally:
        env.close()

    trajectory_hash = write_episode(args.output, episode)
    metrics = compute_metrics(episode, source_trajectory_sha256=trajectory_hash)
    write_metrics(args.output, metrics)
    _print_json(
        {
            "environment_id": args.env_id,
            "policy_id": metadata.policy_id,
            "episode_length": metrics.episode_length,
            "episode_return": metrics.episode_return,
            "task_success": metrics.task_success,
            "output": str(args.output),
            "trajectory_sha256": trajectory_hash,
        }
    )
    return 0


def _run_build_policy(args: argparse.Namespace) -> int:
    metadata = build_lqr_artifact(args.config, args.output)
    _print_json({"policy_id": metadata.policy_id, "candidate": str(args.output)})
    return 0


def _run_train_policy(args: argparse.Namespace) -> int:
    metadata = train_learned_artifact(args.config, args.output)
    _print_json({"policy_id": metadata.policy_id, "candidate": str(args.output)})
    return 0


def _run_freeze_policy(args: argparse.Namespace) -> int:
    suite = load_nominal_suite(args.suite)
    passed = freeze_candidate(args.artifact, suite)
    _print_json({"artifact": str(args.artifact), "frozen": passed})
    return 0 if passed else 1


def _run_evaluate_nominal(args: argparse.Namespace) -> int:
    suite = load_nominal_suite(args.suite)
    summary = evaluate_nominal_suite(
        suite, artifact_root=args.artifact_root, output_dir=args.output
    )
    _print_json(summary)
    return (
        0
        if all(item["quality_gate_passed"] for item in summary["policies"].values())
        else 1
    )


def _run_render_policy(args: argparse.Namespace) -> int:
    policy, metadata = load_policy_artifact(args.artifact)
    env, resolved = make_artifact_environment(metadata, render_mode="human")
    try:
        episode = collect_episode(
            env,
            policy,
            policy_artifact=metadata,
            nominal_environment=metadata.design_spec.environment,
            resolved_environment=resolved,
            episode_seed=args.seed,
        )
    finally:
        env.close()
    metrics = compute_metrics(episode, source_trajectory_sha256="rendered-not-saved")
    _print_json(
        {
            "policy_id": metadata.policy_id,
            "seed": args.seed,
            "episode_length": metrics.episode_length,
            "episode_return": metrics.episode_return,
            "task_success": metrics.task_success,
        }
    )
    return 0


def _run_collect_sweep(args: argparse.Namespace) -> int:
    suite = load_sweep_suite(args.suite)
    result = collect_sweep(
        suite, artifact_root=args.artifact_root, output_dir=args.output
    )
    _print_json(result)
    return 0


def _run_select_boundary_sweep(args: argparse.Namespace) -> int:
    result = plan_boundary_stage(
        args.pilot,
        args.evaluation,
        stage=args.stage,
        output=args.output,
    )
    _print_json(result)
    return 0


def _run_analyze_sweep(args: argparse.Namespace) -> int:
    summary = analyze_sweep(args.evaluation)
    _print_json(
        {
            "suite_id": summary["suite_id"],
            "metric_version": summary["metric_version"],
            "episode_count": summary["episode_count"],
        }
    )
    return 0


def _run_plot_sweep(args: argparse.Namespace) -> int:
    paths = plot_sweep(args.evaluation, args.output)
    _print_json({"figures": [str(path) for path in paths]})
    return 0


def _run_animate_sweep(args: argparse.Namespace) -> int:
    paths = animate_cartpole_sweep(
        args.evaluation,
        args.output,
        layout=args.layout,
        frame_stride=args.frame_stride,
        condition_ids=_split_identifiers(args.condition, "--condition"),
        policy_ids=_split_identifiers(args.policy, "--policy"),
    )
    _print_json({"animation_artifacts": [str(path) for path in paths]})
    return 0


def _split_identifiers(values: list[str] | None, flag: str) -> list[str] | None:
    """Flatten repeated and comma-separated identifier flags."""

    if values is None:
        return None
    identifiers = [
        item.strip() for value in values for item in value.split(",") if item.strip()
    ]
    if not identifiers:
        raise ValueError(f"{flag} requires at least one identifier.")
    return identifiers


def _run_cartpole_symmetry(args: argparse.Namespace) -> int:
    suite = load_cartpole_symmetry_suite(args.suite)
    result = run_cartpole_symmetry_study(
        suite,
        artifact_root=args.artifact_root,
        source_evaluation=args.source_evaluation,
        output_dir=args.output,
        figure_path=args.figure,
    )
    _print_json(
        {
            "suite_id": result["suite_id"],
            "audit_version": result["audit"]["audit_version"],
            "mirror_episode_count": result["mirror_pairs"]["episode_count"],
            "causal_episode_count": result["causal_intervention"]["episode_count"],
            "figure": result["figure"],
        }
    )
    return 0


def _add_config_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def _load_default_nominal(env_id: str):
    filename = NOMINAL_CONFIGS[env_id]
    return load_nominal_env_spec(REPOSITORY_ROOT / "configs" / "envs" / filename)


def _builtin_artifact(nominal: Any) -> PolicyArtifactMetadata:
    design = PolicyDesignSpec(
        schema_version=1,
        design_id="constant-zero-design-v1",
        policy_id=CONSTANT_POLICY_ID,
        policy_type="builtin",
        algorithm="constant-zero",
        algorithm_library="active-eval-gym",
        environment=nominal,
        environment_package_versions=package_versions(nominal.environment_id),
        observation_adapter=IDENTITY_OBSERVATION,
        training_seed=None,
        training_steps=None,
        device=None,
        hyperparameters={"action": "zero"},
    )
    return PolicyArtifactMetadata(
        schema_version=1,
        policy_id=CONSTANT_POLICY_ID,
        design_spec=design,
        artifact_format="builtin-v1",
        model_filename="builtin",
        model_sha256="builtin",
        package_versions={"active-eval-gym": version("active-eval-gym")},
        source_version={"package_version": version("active-eval-gym")},
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
