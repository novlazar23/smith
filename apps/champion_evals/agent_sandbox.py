"""Agent-Sandbox: Jail, Smoke-Test und Adapter für LLM-generierte Agenten-Logik.

Stufe 2 der Agent-Evolution: Der LLM liefert eine einzelne Funktion
``predict(open, high, low, close, volume) -> (p_up, p_down, p_range)``.
Dreischichtige Sicherheit:

1. **Jail** (``validate_agent_code``): statischer AST-Check — nur
   ``numpy``/``math``/``typing``-Imports, exakt eine Funktion ``predict``
   auf Modul-Ebene (keine Modulebene-Aussagen), keine
   erlaubnissensitiven Aufrufe (``eval``/``exec``/``open``/``__import__``/
   ``getattr``/numpy-Dateizugriffe/…).
2. **Eingeschränkte Ausführung** (``load_predict_fn``): ``exec`` in einem
   Namespace mit nur ``np``, ``math`` und einer expliziten Builtin-
   Allowlist — ``open``/``__import__``/``eval`` sind auch zur Laufzeit
   unerreichbar (nicht nur per AST-Check verboten).
3. **Smoke-Test** (``smoke_test_predict``): der generierte Code läuft auf
   synthetischen Fenstern (Volllänge + Warmup-Kurzform); ungültige
   Ausgaben (nicht-finite, Summe ≤ 0) und zu langsame Funktionen
   (``max_ms`` pro Aufruf) werden verworfen.

Der ``EvolvedAgent``-Adapter macht die validierte Funktion zu einem
``BaseAgent``, damit der gesamte bestehende Scoring-/Replay-/
Ensemble-Mechanismus (``replay_instances``, ``score_window``,
``build_ensemble``) unverändert andocken kann. ``load_evolved_agents``
ist fail-soft (fehlende/defekte Datei = leerer Satz, keine Exception).
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from packages.agents.base import AgentConfig, AgentType, BaseAgent
from packages.schemas.agent_report import AgentReport, AgentStatus

logger = logging.getLogger(__name__)

#: Gültiger Agent-Name (snake_case, 3-41 Zeichen).
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
#: Import-Toplevels, die generierter Code verwenden darf.
ALLOWED_IMPORT_TOPS: frozenset[str] = frozenset({"numpy", "math", "typing"})
#: Erlaubnissensitive Namen: dürfen in keinem generierten Code aufgerufen werden.
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
        "getattr",
        "setattr",
        "delattr",
        # numpy-Datei-/Pufferzugriffe (Attribute aufrufbar als np.<name>)
        "fromfile",
        "loadtxt",
        "genfromtxt",
        "frombuffer",
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "load",
    }
)
#: Builtin-Allowlist für den Exec-Namespace (alles andere ist NameError).
_BUILTIN_NAMES: tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "bool",
    "divmod",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "max",
    "min",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


def _safe_import(
    name: str,
    # Drop-in-Ersatz für ``builtins.__import__``: die Parameternamen müssen
    # mit dem CPython-Import-Protokoll übereinstimmen.
    globals: Mapping[str, object] | None = None,  # noqa: A002
    locals: Mapping[str, object] | None = None,  # noqa: A002
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> object:
    """``__import__`` für den Exec-Namespace, beschränkt auf die Import-Allowlist.

    Zweite Schicht hinter dem AST-Jail: auch zur Laufzeit sind
    ``import os``/``from subprocess import ...`` ein ImportError.
    """
    if level or name.split(".")[0] not in ALLOWED_IMPORT_TOPS:
        raise ImportError(f"Import {name!r} ist im Agent-Code verboten")
    return builtins.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS: dict[str, Any] = {name: getattr(builtins, name) for name in _BUILTIN_NAMES}
_SAFE_BUILTINS["__import__"] = _safe_import

#: Obergrenze der Laufzeit pro predict()-Aufruf (ms); der Smoke-Test
#: misst sie, damit ein täglicher Replay (~20k Aufrufe) nicht läuft.
DEFAULT_MAX_MS = 25.0


# ─── Jail (statischer AST-Check) ─────────────────────────────────────────────


def _import_problem(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORT_TOPS:
                return f"Import {alias.name!r} nicht erlaubt"
    else:
        top = (node.module or "").split(".")[0]
        if node.level or top not in ALLOWED_IMPORT_TOPS:
            return f"Import {node.module!r} nicht erlaubt"
    return None


def _forbidden_call(name: str) -> str | None:
    if name in FORBIDDEN_CALL_NAMES:
        return f"Aufruf {name!r} ist im Agent-Code verboten"
    return None


def validate_agent_code(code: str, name: str) -> str | None:
    """Statischer Jail-Check. Returns Ablehnungs-Grund oder ``None``."""
    if not code or not code.strip():
        return "leerer Code"
    if not NAME_PATTERN.match(name):
        return f"Agent-Name {name!r} ungültig"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntaxfehler: {exc.msg} (Zeile {exc.lineno})"

    functions: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problem = _import_problem(node)
            if problem:
                return problem
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # Docstring
        elif isinstance(node, ast.FunctionDef) and node.name == "predict":
            functions.append(node)
        else:
            return f"nicht erlaubtes Modul-Element: {type(node).__name__}"
    if len(functions) != 1:
        return f"erwartet genau 1 Funktion 'predict', gefunden {len(functions)}"

    func = functions[0]
    args = func.args
    if (
        len(args.args) != 5
        or args.vararg
        or args.kwarg
        or args.posonlyargs
        or args.kwonlyargs
    ):
        return "predict(open, high, low, close, volume) erwartet exakt 5 Positional-Parameter"

    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            call_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if call_name:
                problem = _forbidden_call(call_name)
                if problem:
                    return problem
    return None


# ─── Eingeschränkte Ausführung ───────────────────────────────────────────────


def load_predict_fn(code: str, name: str) -> Callable[..., tuple[float, float, float]]:
    """Führt validierten Code in einem eingeschränkten Namespace aus.

    Returns die Funktion ``predict``. Der Namespace enthält nur ``np``,
    ``math`` und die Builtin-Allowlist — Datei-/Import-/Eval-Zugriffe
    sind zur Laufzeit NameError (zweite Schicht hinter dem Jail).

    Raises:
        ValueError: ``predict`` wird nicht definiert.
        Exception: jeder Laufzeitfehler des generierten Codes (Syntax,
            NameError, …) — der Caller entscheidet fail-closed.
    """
    namespace: dict[str, Any] = {
        "__name__": f"_evolved_agent_{name}",
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "math": math,
    }
    exec(compile(code, f"<evolved-agent:{name}>", "exec"), namespace)
    fn = namespace.get("predict")
    if not callable(fn):
        raise ValueError(f"predict() nicht im Namespace gefunden (Name {name!r})")
    return cast("Callable[..., tuple[float, float, float]]", fn)


# ─── Smoke-Test (isolierte Ausführung auf synthetischen Fenstern) ────────────


def _synthetic_window(n: int, seed: int = 7) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Deterministische synthetische Kerzen (Random-Walk). Returns (O,H,L,C,V)."""
    rng = np.random.default_rng(seed)
    price = 100.0
    open_, high, low, close, volume = [], [], [], [], []
    for _ in range(n):
        open_price = price
        close_price = max(1.0, price * (1.0 + rng.normal(0.0, 0.004)))
        high.append(max(open_price, close_price) * (1.0 + abs(rng.normal(0.0, 0.001))))
        low.append(min(open_price, close_price) * (1.0 - abs(rng.normal(0.0, 0.001))))
        open_.append(open_price)
        close.append(close_price)
        volume.append(float(rng.uniform(100.0, 1000.0)))
        price = close_price
    return (
        np.asarray(open_, dtype=np.float64),
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
        np.asarray(volume, dtype=np.float64),
    )


def _check_output(result: object) -> str | None:
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        return "predict() muss ein 3er-Tupel (p_up, p_down, p_range) liefern"
    values: list[float] = []
    for value in result:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return f"Wert {value!r} ist keine Zahl"
        if not math.isfinite(converted):
            return f"Wert {value!r} ist nicht endlich"
        values.append(converted)
    if sum(max(0.0, v) for v in values) <= 0.0:
        return "Wahrscheinlichkeitssumme nach Clipping ≤ 0 (kein Signal)"
    return None


def smoke_test_predict(
    predict_fn: Callable[..., tuple[float, float, float]],
    *,
    max_ms: float = DEFAULT_MAX_MS,
) -> str | None:
    """Treibt ``predict_fn`` über synthetische Fenster (200 + 31 Kerzen).

    Returns Ablehnungs-Grund oder ``None``. Misst zusätzlich die
    Laufzeit: über ``max_ms`` pro Aufruf ist der Kandidat zu langsam für
    den täglichen Replay und wird verworfen.
    """
    for label, window in (("200er-Fenster", _synthetic_window(200)), ("31er-Fenster", _synthetic_window(31))):
        start = time.perf_counter()
        try:
            result = predict_fn(*window)
        except Exception as exc:
            return f"Smoke-Test ({label}) fehlgeschlagen: {type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > max_ms:
            return f"predict() zu langsam: {elapsed_ms:.1f} ms > {max_ms:g} ms im {label}"
        problem = _check_output(result)
        if problem:
            return f"Smoke-Test ({label}): {problem}"
    return None


# ─── Normalisierung und BaseAgent-Adapter ────────────────────────────────────


def normalize_raw_probs(raw: object) -> dict[str, float]:
    """Normalisiert rohe predict()-Werte auf eine gültige Verteilung.

    Negative Werte werden auf 0 geclippt, die Summe auf 1.0 skaliert
    (die Schema-Validierung verlangt 1.0 ± 0.0001; der Rest landet bei
    ``range``).

    Raises:
        ValueError: keine 3 Werte, nicht-finite Werte oder Summe ≤ 0.
    """
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise ValueError(f"predict() muss 3 Werte liefern, bekam {type(raw).__name__}")
    values: list[float] = []
    for value in raw:
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"predict()-Wert {value!r} ist keine Zahl") from exc
        if not math.isfinite(converted):
            raise ValueError(f"predict()-Wert {value!r} ist nicht endlich")
        values.append(max(0.0, converted))
    total = sum(values)
    if total <= 0.0:
        raise ValueError("predict() lieferte keine positive Wahrscheinlichkeitssumme")
    p_up = round(values[0] / total, 6)
    p_down = round(values[1] / total, 6)
    p_range = max(0.0, round(1.0 - p_up - p_down, 6))
    return {"up": p_up, "down": p_down, "range": p_range}


class EvolvedAgent(BaseAgent):
    """``BaseAgent``-Adapter um eine im Jail validierte ``predict``-Funktion."""

    def __init__(
        self,
        config: AgentConfig,
        predict_fn: Callable[..., tuple[float, float, float]],
    ) -> None:
        super().__init__(config)
        self._predict_fn = predict_fn

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        probabilities = normalize_raw_probs(
            self._predict_fn(data["open"], data["high"], data["low"], data["close"], data["volume"])
        )
        dominant = max(probabilities, key=lambda key: probabilities[key])
        direction = {"up": "positive", "down": "negative"}.get(dominant, "neutral")
        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.now(UTC),
            hypothesis=f"Evolved agent (LLM-Logik): {dominant} dominant",
            probabilities=probabilities,
            expected_return=None,
            calibrated_confidence=0.0,
            evidence=[
                self._make_evidence(
                    f"predict:{dominant}",
                    f"p_{dominant}={probabilities[dominant]:.3f}",
                    direction,
                    probabilities[dominant],
                )
            ],
            raw_confidence=probabilities[dominant],
            status=self.config.status,
        )


# ─── Artefakt-IO (evolved_agents.json) ───────────────────────────────────────


def load_evolved_agents(path: Path | str) -> dict[str, dict[str, Any]]:
    """Lädt ``evolved_agents.json`` fail-soft: ``{name: entry}``.

    Fehlende Datei = ``{}``. Defekte Datei (kein JSON-Objekt) = ``{}``
    mit Warning. Einträge ohne gültigen Namen oder ohne ``code``-String
    werden mit Warning übersprungen — diese Funktion wirft nie gegen
    den Aufrufer aus (der Live-Betrieb darf an einer defekten Datei
    nicht scheitern).
    """
    file = Path(path)
    if not file.exists():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("evolved_agents.json nicht lesbar (%s) — ohne Evolved Agents", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("evolved_agents.json erwartet ein JSON-Objekt: %s", file)
        return {}
    agents: dict[str, dict[str, Any]] = {}
    for name, entry in data.items():
        if not isinstance(entry, Mapping):
            logger.warning("evolved_agents.json: Eintrag %r kein Objekt — übersprungen", name)
            continue
        if not NAME_PATTERN.match(str(name)) or not isinstance(entry.get("code"), str):
            logger.warning("evolved_agents.json: Eintrag %r ohne gültigen Namen/Code — übersprungen", name)
            continue
        agents[str(name)] = dict(entry)
    return agents


def build_evolved_agent(
    name: str,
    entry: Mapping[str, Any],
    status: AgentStatus,
    *,
    instrument: str = "",
    horizon: str = "15m",
) -> BaseAgent | None:
    """``EvolvedAgent``-Instanz aus einem Artefakt-Block (``None`` bei Fehler).

    Der Code wird bei jedem Build erneut statisch validiert (billig) und
    in der Sandbox neu ausgeführt — ein korruptes Artefakt kann damit
    keinen Agenten in das Ensemble schmuggeln. Kein Smoke-Test hier:
    der erfolgte bei der Zulassung; Builds laufen pro Zyklus.
    """
    code = entry.get("code")
    if not isinstance(code, str):
        return None
    problem = validate_agent_code(code, name)
    if problem is not None:
        logger.warning("Evolved Agent %r verworfen: %s", name, problem)
        return None
    try:
        predict_fn = load_predict_fn(code, name)
    except Exception as exc:
        logger.warning("Evolved Agent %r nicht ausführbar: %s", name, exc)
        return None
    try:
        status = AgentStatus(str(status).lower())
    except ValueError:
        logger.warning("Evolved Agent %r verworfen: ungültiger Status %r", name, status)
        return None
    version = entry.get("version", 1)
    config = AgentConfig(
        agent_id=name,
        agent_type=AgentType.INDICATOR,
        agent_version=str(version),
        instrument=instrument,
        horizon=horizon,
        status=status,
    )
    return EvolvedAgent(config=config, predict_fn=predict_fn)


def agents_by_code(agents: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Reduziert Artefakt-Einträge auf ``{name: code}`` (für ``build_ensemble``)."""
    return {
        name: str(entry["code"])
        for name, entry in agents.items()
        if isinstance(entry.get("code"), str)
    }
