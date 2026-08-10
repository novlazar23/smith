from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_bootstrap_help_is_side_effect_free(tmp_path: Path) -> None:
    """Help must document bootstrap modes without creating local state."""
    repository_root = Path(__file__).resolve().parents[1]
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    bootstrap = scripts_dir / "bootstrap.sh"
    bootstrap.write_bytes((repository_root / "scripts/bootstrap.sh").read_bytes())
    bootstrap.chmod(0o755)
    (tmp_path / ".env.example").write_text("APP_ENV=development\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    environment = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(bootstrap), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert not (tmp_path / ".env").exists()
