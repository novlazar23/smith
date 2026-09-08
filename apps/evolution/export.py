"""Export des Evolutions-State auf den Host (einseitig: Container → Host).

Der State lebt im Container auf dem ``backtest_reports``-Volume; der Host
bekommt über den Nightly-Export eine kopierte Sicht (``./evolution``):

- ``state/`` — hypotheses/registry/graveyard/state (JSONL/JSON)
- ``strategies/`` — Quelltexte der promoted Mechanismen
- ``registry_patch.txt`` — die zwei Registry-Zeilen pro Mechanismus
  (Adoption in den Host-Tree ist eine bewusste Host-Entscheidung mit
  pyright/ruff/pytest-Gates; der Export macht das **nicht** automatisch).
- ``digest.md`` — aktueller Digest.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .digest import build_digest, fetch_live_paper
from .state import EvolutionStore

logger = logging.getLogger(__name__)

STATE_FILES: tuple[str, ...] = ("hypotheses.jsonl", "graveyard.jsonl", "registry.json", "state.json")


def export_state(store: EvolutionStore, out_dir: Path) -> list[Path]:
    """Kopiert State, Mechanismus-Code, Registry-Patch und Digest nach ``out_dir``.

    Returns die geschriebenen Pfade (leer, wenn nichts zu exportieren ist
    und der Zielordner nicht angelegt werden kann).
    """
    written: list[Path] = []
    try:
        state_dir = out_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        for name in STATE_FILES:
            source = store.root / name
            if source.exists():
                shutil.copy2(source, state_dir / name)
                written.append(state_dir / name)

        for name in sorted(store.promoted_strategy_names()):
            if store.registry()["promoted"] and any(
                item.get("kind") == "mechanism" and item.get("variant", {}).get("strategy") == name
                for item in store.registry().get("promoted", [])
            ):
                source = Path.cwd() / "packages" / "strategies" / f"{name}.py"
                if source.exists():
                    target = out_dir / "strategies" / f"{name}.py"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    written.append(target)

        patches = _registry_patches(store)
        if patches:
            patch_path = out_dir / "registry_patch.txt"
            patch_path.write_text(patches, encoding="utf-8")
            written.append(patch_path)

        digest_path = out_dir / "digest.md"
        digest_path.write_text(build_digest(store, live=fetch_live_paper()), encoding="utf-8")
        written.append(digest_path)
    except OSError as exc:
        logger.warning("Export nach %s fehlgeschlagen: %s", out_dir, exc)
    return written


def _registry_patches(store: EvolutionStore) -> str:
    """Die Registry-Zeilen aller promoted Mechanismen (für die Host-Adoption)."""
    import ast

    lines: list[str] = []
    for item in store.registry().get("promoted", []):
        if item.get("kind") != "mechanism":
            continue
        strategy = item.get("variant", {}).get("strategy", "")
        code = item.get("variant", {}).get("code", "")
        if not strategy or not code:
            continue
        try:
            class_name = next(node.name for node in ast.parse(code).body if isinstance(node, ast.ClassDef))
        except (SyntaxError, StopIteration):
            continue
        lines.append(f"from .{strategy} import {class_name}   # registry.py, Import-Block")
        lines.append(f"    {class_name},   # registry.py, _STRATEGY_CLASSES")
    return "\n".join(lines) + ("\n" if lines else "")
