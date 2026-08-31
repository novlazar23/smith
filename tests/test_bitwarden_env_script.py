import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bitwarden-env.sh"


def test_bitwarden_script_help_needs_no_session() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pull --force" in result.stdout
    assert "BW_SESSION" in result.stdout


def test_session_file_is_git_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", ".bw-session"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ".bw-session"
