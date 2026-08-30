"""CLI for encrypted state transfer and optional Git publication."""

from __future__ import annotations

import argparse
import subprocess

from trading_harness.config import Settings
from trading_harness.services.state_handoff import HandoffCoordinator


def _coordinator(settings: Settings) -> HandoffCoordinator:
    password = settings.state_handoff_password.get_secret_value()
    if not password or not settings.state_node_id:
        raise SystemExit("STATE_HANDOFF_PASSWORD and STATE_NODE_ID must be set in .env")
    return HandoffCoordinator(
        settings.state_data_dir,
        settings.state_bundle_path,
        password,
        settings.state_node_id,
        lease_seconds=settings.state_handoff_lease_seconds,
    )


def _publish(bundle_path: str) -> None:
    subprocess.run(["git", "add", "--", bundle_path], check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", bundle_path], check=False
    ).returncode
    if changed:
        subprocess.run(
            ["git", "commit", "-m", "chore(state): publish encrypted runtime handoff"],
            check=True,
        )
        subprocess.run(["git", "push", "--set-upstream", "origin", "HEAD"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("hand-on", "hand-off", "renew"))
    parser.add_argument("--push", action="store_true", help="commit and push the encrypted bundle")
    args = parser.parse_args()
    settings = Settings()
    coordinator = _coordinator(settings)
    manifest = getattr(coordinator, args.action.replace("-", "_"))()
    if args.push:
        _publish(settings.state_bundle_path)
    print(manifest.model_dump_json(indent=2, exclude={"files"}))


if __name__ == "__main__":
    main()
