import hashlib
import json
from pathlib import Path

import pytest

from active_eval_gym.cli import main
from active_eval_gym.serialization import (
    METADATA_FILENAME,
    METRICS_FILENAME,
    TRAJECTORY_FILENAME,
)


def test_rollout_cli_writes_complete_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "demo"

    result = main(
        [
            "rollout",
            "--env",
            "CartPole-v1",
            "--seed",
            "123",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    metadata = json.loads((output / METADATA_FILENAME).read_text())
    metrics = json.loads((output / METRICS_FILENAME).read_text())
    trajectory = (output / TRAJECTORY_FILENAME).read_bytes()
    assert summary["environment_id"] == "CartPole-v1"
    assert metadata["policy_artifact"]["policy_id"] == "constant-zero-v1"
    assert metadata["schema_version"] == 3
    assert metrics["source_trajectory_sha256"] == hashlib.sha256(trajectory).hexdigest()


def test_rollout_cli_refuses_existing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    output.mkdir()
    (output / METADATA_FILENAME).write_text("existing")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "rollout",
                "--env",
                "CartPole-v1",
                "--seed",
                "123",
                "--output",
                str(output),
            ]
        )

    assert error.value.code == 2
