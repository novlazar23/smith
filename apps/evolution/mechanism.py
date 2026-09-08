"""Mechanismus-Pfad: generierten Strategie-Code im Jail halten.

Pipeline für einen Mechanismus-Vorschlag (LLM-Code):

1. **Jail** (``validate_mechanism_code``): statischer AST-Check —
   exakt eine ``RuleStrategy``-Subklasse, nur erlaubte Imports
   (``typing``/``numpy``/``packages``/relativ), keine erlaubnissensitiven
   Aufrufe (``eval``/``exec``/``open``/``__import__``/…), nur
   Imports/Docstring/Klasse auf Modul-Ebene. Lookahead ist durch die
   300-Bar-Fenster-Semantik der ``RuleStrategy`` ohnehin ausgeschlossen.
2. **Smoke-Test** (``smoke_test``): der Code wird in einem isolierten
   Modul-Namespace ausgeführt (kein Write-Auftritt), mit Default- und
   Grenz-Parametern instanziiert und über 400 synthetische Kerzen
   durchgetrieben.
3. **Registry-Patch** (``apply_registry``/``revert_registry``): zwei
   Zeilen in ``packages/strategies/registry.py`` (Import + Tuple-Eintrag),
   idempotent, revertierbar.
4. **Clone-Check** (``find_clone``/``run_clone_check``): Signal-Zeitpunkte
   des Kandidaten gegen alle Zoo-Strategien auf derselben Sample-Periode —
   > 90 % Überlappung (Containment in mindestens einer Richtung) bedeutet
   „verkleidete Konfig-Variante", kein neuer Mechanismus.

Abgelehnte Mechanismen: Datei + Registry-Zeilen werden entfernt, der
Ablehnungsgrund landet im Grab.
"""

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import numpy as np
from packages.backtesting.core import Candle
from packages.backtesting.strategies import StrategySignal
from packages.strategies import STRATEGIES, create_strategy

from .models import NAME_PATTERN

logger = logging.getLogger(__name__)

#: Import-Toplevels, die generierter Code verwenden darf.
ALLOWED_IMPORT_TOPS: frozenset[str] = frozenset({"typing", "numpy", "packages"})
#: Erlaubnissensitive Namen, die in keinem generierten Code auftauchen dürfen.
FORBIDDEN_CALL_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "breakpoint",
        "input",
        "vars",
        "globals",
        "locals",
        "memoryview",
        "type",
    }
)
#: Clone-Check: Overlap-Schwelle (Containment in min. einer Richtung).
CLONE_THRESHOLD = 0.9
#: Fensterlänge der RuleStrategy (maximal sinnvoller Warmup).
MAX_WINDOW = 300


# ─── Jail (statischer AST-Check) ────────────────────────────────────────────


def _import_problem(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORT_TOPS:
                return f"Import {alias.name!r} nicht erlaubt"
    else:
        if node.level and node.level > 1:
            return f"relativer Import zu tief (level {node.level})"
        if not node.level:
            top = (node.module or "").split(".")[0]
            if top not in ALLOWED_IMPORT_TOPS:
                return f"Import {node.module!r} nicht erlaubt"
    return None


def _forbidden_call(name: str) -> str | None:
    if name in FORBIDDEN_CALL_NAMES:
        return f"Aufruf {name!r} ist im Mechanismus-Code verboten"
    return None


def validate_mechanism_code(code: str, strategy_name: str) -> str | None:
    """Statischer Jail-Check. Returns Ablehnungs-Grund oder ``None``."""
    if not code or not code.strip():
        return "leerer Code"
    if not NAME_PATTERN.match(strategy_name):
        return f"Strategie-Name {strategy_name!r} ungültig"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntaxfehler: {exc.msg} (Zeile {exc.lineno})"

    classes: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problem = _import_problem(node)
            if problem:
                return problem
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # Docstring
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, ast.Pass):
            continue
        else:
            return f"nicht erlaubtes Modul-Element: {type(node).__name__}"

    if len(classes) != 1:
        return f"erwartet genau 1 Klasse, gefunden {len(classes)}"
    cls = classes[0]

    base_names: list[str] = []
    for base in cls.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)
        else:
            return "ungültige Base-Klasse"
    if base_names != ["RuleStrategy"]:
        return f"exakt eine Base 'RuleStrategy' erwartet, gefunden {base_names}"

    assigned: set[str] = set()
    methods: set[str] = set()
    strategy_name_literal: str | None = None
    min_bars_literal: int | None = None
    for stmt in cls.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
                    if isinstance(stmt.value, ast.Constant):
                        if target.id == "strategy_name" and isinstance(stmt.value.value, str):
                            strategy_name_literal = stmt.value.value
                        if target.id == "min_bars" and isinstance(stmt.value.value, int):
                            min_bars_literal = stmt.value.value
        elif isinstance(stmt, ast.FunctionDef):
            methods.add(stmt.name)
            if stmt.name == "_evaluate" and any(
                isinstance(child, ast.Call)
                and (
                    (isinstance(child.func, ast.Name) and _forbidden_call(child.func.id))
                    or (isinstance(child.func, ast.Attribute) and _forbidden_call(child.func.attr))
                )
                for child in ast.walk(stmt)
            ):
                return "verbotener Aufruf in _evaluate"

    missing_required = {"strategy_name", "description", "param_specs"} - assigned
    if missing_required:
        return f"fehlende ClassVars: {', '.join(sorted(missing_required))}"
    if "_evaluate" not in methods:
        return "Methode _evaluate fehlt"
    if strategy_name_literal is not None and strategy_name_literal != strategy_name:
        return f"strategy_name {strategy_name_literal!r} != erwarteter Name {strategy_name!r}"
    if min_bars_literal is not None and not (1 <= min_bars_literal <= MAX_WINDOW):
        return f"min_bars {min_bars_literal} außerhalb [1, {MAX_WINDOW}]"

    # Verbotene Aufrufe im gesamten Klassen-Körper (auch in Default-Ausdrücken).
    for node in ast.walk(cls):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name:
                problem = _forbidden_call(name)
                if problem:
                    return problem
    return None


# ─── Smoke-Test (isolierte Ausführung) ──────────────────────────────────────


def _synthetic_candles(n: int = 400, start_price: float = 100.0, seed: int = 7) -> list[Candle]:
    """Deterministische synthetische 5m-Kerzen (Random-Walk + Saisonalität)."""
    from datetime import UTC, datetime, timedelta

    rng = np.random.default_rng(seed)
    candles: list[Candle] = []
    price = start_price
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
        drift = 0.001 * np.sin(i / 40.0)
        shock = rng.normal(0.0, 0.004)
        open_price = price
        close_price = max(1.0, price * (1.0 + drift + shock))
        high = max(open_price, close_price) * (1.0 + abs(rng.normal(0.0, 0.001)))
        low = min(open_price, close_price) * (1.0 - abs(rng.normal(0.0, 0.001)))
        candles.append(
            Candle(
                timestamp=base + timedelta(minutes=5 * i),
                symbol="SYNTH/USDT",
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=float(rng.uniform(100.0, 1000.0)),
            )
        )
        price = close_price
    return candles


def _load_class_from_code(code: str, strategy_name: str) -> type:
    """Führt den Code in einem isolierten Modul aus und liefert die Klasse.

    Der Modulname liegt unter ``packages.strategies`` (relativer Import
    ``from . import indicators`` funktioniert so), der Eintrag in
    ``sys.modules`` wird danach wieder entfernt.
    """
    module_name = f"packages.strategies._evo_jail_{strategy_name}"
    module = types.ModuleType(module_name)
    module.__package__ = "packages.strategies"
    module.__file__ = f"packages/strategies/{strategy_name}.py"
    sys.modules[module_name] = module
    try:
        exec(compile(code, f"<mechanism:{strategy_name}>", "exec"), module.__dict__)
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == module_name:
                return value
    finally:
        sys.modules.pop(module_name, None)
    raise ValueError("Klasse nicht im Modul gefunden")


def smoke_test(code: str, strategy_name: str) -> str | None:
    """Instanziert (Default + Parameter-Grenzen) und treibt 400 Kerzen durch.

    Returns Ablehnungs-Grund oder ``None``.
    """
    try:
        cls = _load_class_from_code(code, strategy_name)
    except Exception as exc:
        return f"Code nicht ausführbar: {type(exc).__name__}: {exc}"
    try:
        spec = cls.param_specs or {}
        configs: list[dict[str, float]] = [{}]
        for key, (_, lo, hi) in spec.items():
            configs.append({key: float(lo)})
            configs.append({key: float(hi)})
        candles = _synthetic_candles()
        for params in configs:
            strategy = cls("SYNTH/USDT", **params)
            for candle in candles:
                signal = strategy.on_bar(candle)
                if signal is not None and not isinstance(signal, StrategySignal):
                    return f"on_bar liefert {type(signal).__name__} statt StrategySignal|None"
    except Exception as exc:
        return f"Smoke-Test fehlgeschlagen: {type(exc).__name__}: {exc}"
    return None


# ─── Datei- und Registry-Verwaltung ─────────────────────────────────────────


def write_strategy_file(code: str, strategy_name: str, repo_root: Path) -> Path:
    """Schreibt ``packages/strategies/<name>.py`` (idempotent)."""
    path = Path(repo_root) / "packages" / "strategies" / f"{strategy_name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code.rstrip() + "\n", encoding="utf-8")
    return path


def _class_name_from_code(code: str, strategy_name: str) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name
    raise ValueError(f"Klasse nicht gefunden (Name {strategy_name!r})")


def apply_registry(code: str, strategy_name: str, repo_root: Path) -> str:
    """Trägt die Strategie in ``registry.py`` ein (idempotent). Returns Klassenname."""
    class_name = _class_name_from_code(code, strategy_name)
    registry_path = Path(repo_root) / "packages" / "strategies" / "registry.py"
    text = registry_path.read_text(encoding="utf-8")

    import_line = f"from .{strategy_name} import {class_name}\n"
    tuple_line = f"    {class_name},\n"
    if import_line not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from .") and "import" in line:
                insert_at = i + 1
        lines.insert(insert_at, import_line)
        text = "".join(lines)
    if tuple_line not in text:
        marker = "tuple[type[RuleStrategy], ...] = (\n"
        if marker not in text:
            raise ValueError("_STRATEGY_CLASSES-Marker nicht in registry.py gefunden")
        text = text.replace(marker, marker + tuple_line, 1)
    registry_path.write_text(text, encoding="utf-8")
    return class_name


def revert_registry(code: str, strategy_name: str, repo_root: Path) -> None:
    """Entfernt die zwei Registry-Zeilen (best-effort; Dateilöschung durch Caller)."""
    try:
        class_name = _class_name_from_code(code, strategy_name)
    except (ValueError, SyntaxError):
        return
    registry_path = Path(repo_root) / "packages" / "strategies" / "registry.py"
    text = registry_path.read_text(encoding="utf-8")
    import_line = f"from .{strategy_name} import {class_name}\n"
    tuple_line = f"    {class_name},\n"
    text = text.replace(import_line, "").replace(tuple_line, "")
    registry_path.write_text(text, encoding="utf-8")


# ─── Clone-Check ─────────────────────────────────────────────────────────────


def signal_bars(strategy_name: str, params: dict[str, float], candles: list[Candle]) -> set[int]:
    """Bar-Indizes, an denen die Strategie ein BUY-Signal feurt."""
    from packages.backtesting.strategies import SignalAction

    strategy = create_strategy(strategy_name, candles[0].symbol, dict(params))
    bars: set[int] = set()
    for index, candle in enumerate(candles):
        signal = strategy.on_bar(candle)
        if signal is not None and signal.action is SignalAction.BUY:
            bars.add(index)
    return bars


def clone_similarity(a: set[int], b: set[int]) -> float:
    """Containment-Überlappung in beiden Richtungen (0, wenn eine Menge leer ist)."""
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    return max(overlap / len(a), overlap / len(b))


def find_clone(candidate: set[int], zoo_signals: dict[str, set[int]], threshold: float = CLONE_THRESHOLD) -> str | None:
    """Erste Zoo-Strategie mit ≥ ``threshold`` Overlap (alphabetisch)."""
    for name in sorted(zoo_signals):
        if clone_similarity(candidate, zoo_signals[name]) >= threshold:
            return name
    return None


def run_clone_check(
    strategy_name: str,
    params: dict[str, float],
    candles: list[Candle],
    *,
    threshold: float = CLONE_THRESHOLD,
) -> str | None:
    """Clone-Check gegen alle Zoo-Strategien (Default-Parameter) auf derselben Periode."""
    if len(candles) < 350:
        logger.info("Clone-Check übersprungen — zu wenige Kerzen (%d)", len(candles))
        return None
    candidate = signal_bars(strategy_name, params, candles)
    zoo = {
        name: signal_bars(name, {}, candles)
        for name in sorted(STRATEGIES)
        if name != strategy_name
    }
    return find_clone(candidate, zoo, threshold)


def sample_clone_candles(candles: list[Candle], days: int = 90) -> list[Candle]:
    """Letzte ``days`` Tage der übergebenen Kerzen (Clone-Check-Periode)."""
    if not candles:
        return []
    from datetime import timedelta

    cutoff = candles[-1].timestamp - timedelta(days=days)
    return [candle for candle in candles if candle.timestamp >= cutoff]
