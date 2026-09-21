"""Stufe 2: Zulassung und Re-Prüfung LLM-generierter Agenten.

Preregistrierte, deterministische Regeln (der LLM kennt sie nicht):

- **ZULASSUNG** (``judge_candidate``): OOS-Score (1 - Brier) ≥
  ``RANDOM_BASELINE_SCORE + ADMISSION_MARGIN`` UND Stabilitäts-Guard
  (OOS-Hit-Rate nicht weiter als ``STABILITY_TOLERANCE`` unter der
  eigenen Kalibrierungs-Hit-Rate) UND positiver LOO-Marginal-Beitrag
  (das bestehende Basis-Ensemble wird durch den Agenten besser, nicht
  nur redundant schlechter).
- **RE-PRÜFUNG** (``judge_retention``): bereits zugelassene Agenten
  werden jeden Lauf auf dem aktuellen Fenster neu bewertet; unter
  ``RANDOM_BASELINE_SCORE`` oder mit nicht-positivem LOO-Marginal-
  Beitrag (oder ohne durchgehendes Delivern) werden sie entfernt.
- **Deckel** (``MAX_EVOLVED_AGENTS``): maximal N zugelassene Agenten
  (höchster OOS-Score bleibt) — der gewichtete Konsens verdünnt sich
  mit jedem zusätzlichen Mitglied.
- **Steigende Hurdle** (``trial_ledger``): Die Zulassungs-Margin wächst
  mit der kumulativen Anzahl geprüfter Kandidaten (Trial-Count aus
  ``evolved_agents_trials.json`` neben ``evolved_agents.json``):
  ``ADMISSION_MARGIN + 0.005 · log2(Trials)`` pro Verdopplung des
  Suchraums — jeder Kandidat prüft dieselbe rollierende OOS-Datenbasis,
  falsche Zulassungen häufen sich mit N (Multiple-Testing-Korrektur).
  Bestand-Agenten (Re-Prüfung) tragen diese Anhebung nicht.

Zugelassene Agenten landen atomar in ``evolved_agents.json``
(Code, Claim, Version, Score); ``build_ensemble`` hängt sie dem
Ensemble als SHADOW-Mitglieder an (siehe ``agent_sandbox``).
"""

from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from packages.llm.client import LLMClient
    from packages.schemas.agent_report import AgentStatus

from packages.agents.base import BaseAgent
from packages.backtesting.core import Candle
from packages.validation.target_variables import TargetConfig

from .agent_sandbox import build_evolved_agent, load_evolved_agents, smoke_test_predict
from .score import (
    AgentMetrics,
    EvalSample,
    oos_score_vs_random_base,
    replay_instances,
    score_window,
)
from .sequential_test import daily_lag, holm_reject, one_sided_z_pvalue
from .trial_ledger import (
    TRIALS_FILENAME_EVOLVED,
    admission_margin,
    load_trial_count,
    record_trial_count,
)

logger = logging.getLogger(__name__)

#: Score (1 - Brier) eines uniformen 3-Klassen-Prädiktors:
#: Brier = Σ_c (1/3 - y_c)² = 2/3 (exakt, unabhängig vom Outcome)
#: → Score = 1/3 ≈ 0.3333. Agenten, die das nicht schlagen, sind
#: im Ensemble reines Rauschen.
RANDOM_BASELINE_SCORE = 1.0 - 2.0 / 3.0
#: Minimaler Vorsprung auf die Zufalls-Basis für eine Zulassung.
ADMISSION_MARGIN = 0.02
#: OOS-Hit-Rate darf höchstens so weit unter der Kalibrierungs-Hit-Rate liegen
#: (derselbe Overfitting-Guard wie die Parameter-Evolution).
STABILITY_TOLERANCE = 0.05
#: Maximal zugelassene Evolved Agents im Ensemble.
MAX_EVOLVED_AGENTS = 3
EVOLVED_AGENTS_FILENAME = "evolved_agents.json"
#: Letzter Lauf: Urteile aller geprüften Agenten (Beobachtung/UI).
EVOLVED_AGENTS_LAST_RUN_FILENAME = "evolved_agents_last_run.json"


@dataclass(frozen=True)
class AgentVerdict:
    """Deterministische Beurteilung eines Kandidaten bzw. Bestand-Agenten."""

    name: str
    admitted: bool
    score: float
    reasons: tuple[str, ...]


def judge_candidate(
    name: str,
    metrics: AgentMetrics,
    *,
    promotion_margin: float = ADMISSION_MARGIN,
    stability_tolerance: float = STABILITY_TOLERANCE,
) -> AgentVerdict:
    """Zulassungs-Check (deterministisch, preregistriert)."""
    score = 1.0 - metrics.oos_brier
    reasons: list[str] = []
    if score < RANDOM_BASELINE_SCORE + promotion_margin:
        reasons.append(
            f"OOS-Score {score:.4f} < Zufalls-Basis {RANDOM_BASELINE_SCORE:.4f} + {promotion_margin:g}"
        )
    if metrics.oos_stability < metrics.cal_stability - stability_tolerance:
        reasons.append("Stabilitäts-Guard verletzt (OOS-Hit deutlich unter Kalibrierungs-Hit)")
    if metrics.oos_marginal <= 0.0:
        reasons.append(f"LOO-Marginal-Beitrag {metrics.oos_marginal:+.4f} ≤ 0 (keine zusätzliche Information)")
    return AgentVerdict(name=name, admitted=not reasons, score=score, reasons=tuple(reasons))


def judge_retention(name: str, metrics: AgentMetrics) -> AgentVerdict:
    """Re-Prüfung Bestand-Agenten: bleibt nur, wenn er die Zufalls-Basis
    (ohne Zulassungs-Vorsprung) weiterhin schlägt und positiv zur
    Informationsbasis des Ensembles beiträgt."""
    score = 1.0 - metrics.oos_brier
    reasons: list[str] = []
    if score < RANDOM_BASELINE_SCORE:
        reasons.append(f"OOS-Score {score:.4f} < Zufalls-Basis {RANDOM_BASELINE_SCORE:.4f}")
    if metrics.oos_marginal <= 0.0:
        reasons.append(f"LOO-Marginal-Beitrag {metrics.oos_marginal:+.4f} ≤ 0")
    return AgentVerdict(name=name, admitted=not reasons, score=score, reasons=tuple(reasons))


def prepare_candidates(
    proposals: Sequence[Mapping[str, str]],
    *,
    instrument: str = "",
    horizon: str = "15m",
    summary: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, BaseAgent], dict[str, str], dict[str, str]]:
    """Jail + Smoke-Test + Adapter für LLM-Vorschläge (fail-closed pro Kandidat).

    ``summary`` (optional) erhält für jeden in dieser Phase verworfenen
    Kandidaten ein Urteil (``kind="kandidat"``, ``admitted=False``), damit
    das Letzter-Lauf-Artefakt den vollständigen Weg jedes Vorschlags zeigt.

    Returns:
        (instances, code_by_name, claim_by_name) — nur Kandidaten, die
        Jail, Smoke-Test und Adapter überstanden haben.
    """
    from packages.schemas.agent_report import AgentStatus

    from .agent_sandbox import load_predict_fn, validate_agent_code

    def _rejected(name: str, reason: str) -> None:
        if summary is not None:
            summary.append({"name": name, "kind": "kandidat", "admitted": False, "score": None, "reasons": [reason]})

    instances: dict[str, BaseAgent] = {}
    code_by_name: dict[str, str] = {}
    claim_by_name: dict[str, str] = {}
    for proposal in proposals:
        name, code, claim = proposal["name"], proposal["code"], proposal["claim"]
        problem = validate_agent_code(code, name)
        if problem is not None:
            logger.info("Kandidat %s verworfen (Jail): %s", name, problem)
            _rejected(name, f"Jail: {problem}")
            continue
        try:
            predict_fn = load_predict_fn(code, name)
        except Exception as exc:
            logger.info("Kandidat %s verworfen (Ausführung): %s", name, exc)
            _rejected(name, f"Ausführung: {exc}")
            continue
        problem = smoke_test_predict(predict_fn)
        if problem is not None:
            logger.info("Kandidat %s verworfen (Smoke-Test): %s", name, problem)
            _rejected(name, f"Smoke-Test: {problem}")
            continue
        agent = build_evolved_agent(
            name,
            {"code": code, "version": 1},
            AgentStatus.SHADOW,
            instrument=instrument,
            horizon=horizon,
        )
        if agent is None:
            logger.warning("Kandidat %s verworfen (Adapter fehlgeschlagen)", name)
            _rejected(name, "Adapter fehlgeschlagen")
            continue
        instances[name] = agent
        code_by_name[name] = code
        claim_by_name[name] = claim
    return instances, code_by_name, claim_by_name


def evaluate_evolved_candidates(
    series: list[tuple[str, list[Candle]]],
    base_instances: Mapping[str, BaseAgent],
    candidates: Mapping[str, BaseAgent],
    candidate_meta: Mapping[str, tuple[str, str]],
    *,
    previous: Mapping[str, Mapping[str, Any]],
    candle_limit: int = 200,
    min_candles: int = 30,
    evaluate_every: int = 5,
    horizon_bars: int = 3,
    target_config: TargetConfig | None = None,
    calibration_ratio: float = 0.5,
    max_agents: int = MAX_EVOLVED_AGENTS,
    promotion_margin: float = ADMISSION_MARGIN,
    summary: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Eine Replay-Runde (Basis + Bestand + Kandidaten) → neues Artefakt.

    ``summary`` (optional) erhält pro geprüftem Agenten einen
    Urteils-Eintrag (``name``, ``kind``, ``admitted``, ``score``,
    ``reasons``) — der Aufrufer persistiert sie als Letzter-Lauf-Artefakt.

    Alle Instanzen sehen auf jedem Schritt exakt dasselbe Fenster
    (``replay_instances``); ``score_window`` bewertet nur Agenten, die
    in JEDEM Schritt liefern — ein Kandidat, der auch nur einmal
    fehlschlägt, erhält keine Metriken und wird damit abgelehnt
    (Fail-Closed).

    Args:
        series: (instrument, Kerzen)-Paare für die Replay-Fenster.
        base_instances: das 4er-Basis-Ensemble (Champion-Parameter).
        candidates: neue LLM-Kandidaten (bereits Jailed/Smoke-getestet).
        candidate_meta: ``name → (code, claim)`` für das Artefakt.
        previous: Bestand-``evolved_agents.json`` (``name → entry``).
        promotion_margin: Zulassungs-Margin (Default ``ADMISSION_MARGIN``);
            ``run_agent_evolution`` übergibt die nach Trial-Count
            angehobene Hurdle (``trial_ledger.admission_margin``).

    Returns:
        Der neue ``evolved_agents.json``-Inhalt (ohne Schreib-Zugriff).
    """
    admitted_instances: dict[str, BaseAgent] = {}
    for name, entry in previous.items():
        agent = build_evolved_agent(name, entry, _retention_status())
        if agent is None:
            logger.warning("Bestand-Agent %s nicht wiederbaubar — wird entfernt", name)
            continue
        admitted_instances[name] = agent

    instances: dict[str, BaseAgent] = {
        **dict(base_instances),
        **admitted_instances,
        **dict(candidates),
    }
    samples: list[EvalSample] = []
    for _instrument, candles in series:
        samples.extend(
            replay_instances(
                candles,
                instances,
                candle_limit=candle_limit,
                min_candles=min_candles,
                evaluate_every=evaluate_every,
                horizon_bars=horizon_bars,
                target_config=target_config,
            )
        )
    metrics = score_window(samples, calibration_ratio=calibration_ratio) if samples else {}

    # Shadow-Sequenztest (Stufe 2): einseitiger z-Test (Score - Zufalls-
    # Basis) pro Kandidat + Holm über den Batch — protokolliert parallel
    # zu den Zulassungs-Gates, entscheidet NICHT (s. sequential_test).
    # Fail-soft wie die restlichen Schattenpfade.
    shadow_p: dict[str, float] = {}
    nw_lag: int | None = None
    try:
        for name in candidates:
            delta_series = oos_score_vs_random_base(samples, name, calibration_ratio=calibration_ratio)
            if not delta_series:
                continue
            if nw_lag is None:
                nw_lag = daily_lag([ts for ts, _ in delta_series])
            shadow_p[name] = one_sided_z_pvalue(
                [d for _, d in delta_series], min_effect=ADMISSION_MARGIN, nw_lag=nw_lag
            )
    except Exception:
        logger.warning("Shadow-Sequenztest (Stufe 2) fehlgeschlagen (Zulassung bleibt unberührt)", exc_info=True)
        shadow_p = {}
    shadow_rejected_names = (
        {name for i, name in enumerate(shadow_p) if i in holm_reject(list(shadow_p.values()))} if shadow_p else set()
    )
    if shadow_p:
        logger.info(
            "Shadow-Sequenztest (Stufe 2): %d/%d Kandidaten über Holm alpha=0.05 (NW-Lag=%s)",
            len(shadow_rejected_names),
            len(shadow_p),
            nw_lag,
        )

    current: dict[str, tuple[str, str, float]] = {}  # name → (code, claim, score)

    for name in previous:
        entry = previous[name]
        code, claim = str(entry.get("code", "")), str(entry.get("claim", ""))
        agent_metrics = metrics.get(name)
        if agent_metrics is None:
            logger.info("Bestand-Agent %s entfernt: hat nicht in jedem Schritt geliefert", name)
            if summary is not None:
                summary.append(
                    {"name": name, "kind": "bestand", "admitted": False, "score": None, "reasons": ["hat nicht in jedem Schritt geliefert"]}
                )
            continue
        verdict = judge_retention(name, agent_metrics)
        if verdict.admitted:
            current[name] = (code, claim, verdict.score)
            logger.info("Bestand-Agent %s bestätigt (Score %.4f)", name, verdict.score)
        else:
            logger.info("Bestand-Agent %s entfernt: %s", name, "; ".join(verdict.reasons))
        if summary is not None:
            summary.append(
                {"name": name, "kind": "bestand", "admitted": verdict.admitted, "score": verdict.score, "reasons": list(verdict.reasons)}
            )

    for name in candidates:
        agent_metrics = metrics.get(name)
        if agent_metrics is None:
            logger.info("Kandidat %s abgelehnt: hat nicht in jedem Schritt geliefert", name)
            if summary is not None:
                summary.append(
                    {
                        "name": name,
                        "kind": "kandidat",
                        "admitted": False,
                        "score": None,
                        "reasons": ["hat nicht in jedem Schritt geliefert"],
                        "shadow_p": shadow_p.get(name),
                        "shadow_holm_rejected": name in shadow_rejected_names,
                    }
                )
            continue
        verdict = judge_candidate(name, agent_metrics, promotion_margin=promotion_margin)
        if verdict.admitted:
            code, claim = candidate_meta[name]
            current[name] = (code, claim, verdict.score)
            logger.info(
                "Kandidat %s ZUGELASSEN (OOS-Score %.4f, Zulassungs-Margin %.4f, LOO %.4f, Hit %.3f→%.3f)",
                name,
                verdict.score,
                promotion_margin,
                agent_metrics.oos_marginal,
                agent_metrics.cal_stability,
                agent_metrics.oos_stability,
            )
        else:
            logger.info(
                "Kandidat %s abgelehnt (Zulassungs-Margin %.4f): %s",
                name,
                promotion_margin,
                "; ".join(verdict.reasons),
            )
        if summary is not None:
            summary.append(
                {
                    "name": name,
                    "kind": "kandidat",
                    "admitted": verdict.admitted,
                    "score": verdict.score,
                    "reasons": list(verdict.reasons),
                    "shadow_p": shadow_p.get(name),
                    "shadow_holm_rejected": name in shadow_rejected_names,
                }
            )

    if len(current) > max_agents:
        ranked = sorted(current.items(), key=lambda item: -item[1][2])
        for name, meta in ranked[max_agents:]:
            logger.info("Agent %s entfernt: Deckel %d überschritten (niedrigster Score)", name, max_agents)
            if summary is not None:
                summary.append(
                    {
                        "name": name,
                        "kind": "bestand" if name in previous else "kandidat",
                        "admitted": False,
                        "score": meta[2],
                        "reasons": ["Deckel überschritten (niedrigster Score)"],
                    }
                )
        current = dict(ranked[:max_agents])

    return build_agents_artifact(current, previous)


def _retention_status() -> AgentStatus:
    """Status für Bestand-Agenten im Replay (irrelevant fürs Scoring)."""
    from packages.schemas.agent_report import AgentStatus

    return AgentStatus.SHADOW


def build_agents_artifact(
    current: Mapping[str, tuple[str, str, float]],
    previous: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Neues ``evolved_agents.json``: pro Agent ``version``/``code``/
    ``claim``/``score``/``admitted_at``.

    Unveränderter Code (Bestätigung) behält Block und Version; neu
    zugelassene oder geänderte Agenten bumpt die Version um 1.
    """
    prev: Mapping[str, Mapping[str, Any]] = previous or {}
    artifact: dict[str, dict[str, Any]] = {}
    for name, (code, claim, score) in current.items():
        entry = prev.get(name)
        if entry is not None and entry.get("code") == code:
            artifact[name] = {**dict(entry), "score": score}
        else:
            version = 1 if entry is None else int(entry.get("version", 0)) + 1
            admitted_at = str(entry.get("admitted_at")) if entry is not None else datetime.now(UTC).isoformat(timespec="seconds")
            artifact[name] = {
                "version": version,
                "code": code,
                "claim": claim,
                "score": score,
                "admitted_at": admitted_at,
            }
    return artifact


def build_digest(
    eval_artifact: Mapping[str, Mapping[str, Any]] | None,
    previous_agents: Mapping[str, Mapping[str, Any]],
) -> str:
    """Evidenz-Digest für den Proposer-Aufruf (deterministisch, klein).

    Basis: pro Basis-Agent OOS-Score/Stabilität/Marginal aus dem
    ``champion_evals.json``-Challenger-Block plus bereits zugelassene
    Evolved Agents mit ihren Scores. Fehlt die Datenbasis, sagt der
    Digest das explizit (der LLM soll dann konservative, robuste Logik
    vorschlagen statt nach dem Fenster fitten).
    """
    lines: list[str] = []
    if eval_artifact:
        for agent_id in sorted(eval_artifact):
            challenger = eval_artifact[agent_id].get("challenger") or {}
            if not challenger:
                continue
            lines.append(
                f"- {agent_id}: OOS-Score {challenger.get('oos_score', 0.0):.4f}, "
                f"Stabilität {challenger.get('stability_score', 0.0):.3f}, "
                f"LOO-Marginal {challenger.get('marginal_contribution', 0.0):+.4f}"
            )
    else:
        lines.append("- (noch keine Evaluationsdaten — keine übermäßige Spezialisierung)")
    if previous_agents:
        for name in sorted(previous_agents):
            score = previous_agents[name].get("score", 0.0)
            lines.append(f"- zugelassen {name}: OOS-Score {float(score):.4f} (Version {previous_agents[name].get('version', 1)})")
    else:
        lines.append("- (noch keine zugelassenen Evolved Agents)")
    lines.append(f"Zufalls-Basis (uniformer 3-Klassen-Prädiktor): Score {RANDOM_BASELINE_SCORE:.4f}")
    return "\n".join(lines)


def load_eval_artifact(path: str | Path) -> dict[str, Any] | None:
    """Lädt das ``champion_evals.json``-Artefakt fail-soft (für den Digest)."""
    import json

    file = Path(path)
    if not file.exists():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_candidate_file(path: str | Path, name: str, claim: str) -> list[dict[str, str]]:
    """Lädt einen handgeschriebenen Kandidaten aus einer ``.py``-Datei.

    Die Datei enthält den Kandidaten-Code (derselbe Sandbox-Vertrag wie
    LLM-Vorschläge: nur Import + ``predict``); Name und Claim kommen von
    der CLI, weil der Sandbox-Vertrag keine Modul-Konstanten erlaubt.
    Der Kandidat läuft exakt dieselben Zulassungs-Gates durch wie
    LLM-Vorschläge (Jail, Smoke-Test, Replay, ``judge_candidate``).

    Raises:
        ValueError: Datei nicht lesbar oder leer (fail-closed — ein
            Kandidaten-Lauf ohne Kandidaten wäre stillschweigend ein
            reiner Bestand-Re-Check).
    """
    file = Path(path)
    try:
        code = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Kandidaten-Datei nicht lesbar: {file} ({exc})") from exc
    if not code.strip():
        raise ValueError(f"Kandidaten-Datei leer: {file}")
    return [{"name": name, "claim": claim, "code": code}]


def run_agent_evolution(
    *,
    args: Namespace,
    series: list[tuple[str, list[Candle]]],
    llm_client_factory: Callable[[], Any] | None = None,
) -> int:
    """Kompletter Stufe-2-Lauf: LLM-Vorschläge und/oder handgeschriebene
    Kandidaten (``args.candidate_file``) → Jail/Smoke → Replay →
    Zulassung/Re-Prüfung → atomares Artefakt.

    Handgeschriebene Kandidaten laufen exakt dieselben (preregistrierten)
    Zulassungs-Gates durch wie LLM-Vorschläge — es gibt keinen Sonderweg.

    Fail-soft auf allen Ebenen: kein LLM = keine neuen LLM-Kandidaten
    (Bestand wird trotzdem re-geprüft); fehlende Bestand-Datei = leerer
    Start. Returns 0 bei Durchlauf (auch ohne Zulassungen), 1 nur bei
    Daten-Fehlern (keine Kerzen).

    Args:
        args: CLI-Namespace (evolve_agents, candidate_file, candidate_name,
            candidate_claim, agents_output, output, …).
        series: (instrument, Kerzen)-Paare (gemeinsam mit Stufe 1 geladen).
        llm_client_factory: Factory für den LLM-Client (Tests);
            Default ``LLMClient.from_env``.
    """
    from pathlib import Path

    from .evolve import write_json_atomic

    agents_path = Path(args.agents_output or Path(args.output).with_name(EVOLVED_AGENTS_FILENAME))
    previous = load_evolved_agents(agents_path)

    # Preregistrierte Multi-Testing-Korrektur: Zulassungs-Hurdle steigt
    # mit dem kumulativen Trial-Count (jeder Kandidat prüft dieselbe
    # rollierende OOS-Datenbasis) — siehe trial_ledger.
    trial_path = agents_path.with_name(TRIALS_FILENAME_EVOLVED)
    trials_before = load_trial_count(trial_path)

    base_instances = _base_ensemble(args)
    target = TargetConfig(
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        horizon=args.horizon,
    )

    candidates: dict[str, BaseAgent] = {}
    candidate_meta: dict[str, tuple[str, str]] = {}
    summary: list[dict[str, Any]] = []
    proposals: list[Mapping[str, str]] = []
    candidate_file = getattr(args, "candidate_file", None)
    if candidate_file:
        proposals.extend(
            load_candidate_file(
                candidate_file,
                name=getattr(args, "candidate_name", None) or Path(candidate_file).stem,
                claim=getattr(args, "candidate_claim", None) or "",
            )
        )
    if args.evolve_agents:
        llm_client = llm_client_factory() if llm_client_factory is not None else _default_llm_client(args.llm_model)
        if llm_client is not None:
            from .agent_params import AGENT_TYPES
            from .proposer import propose

            digest = build_digest(load_eval_artifact(args.output), previous)
            taken = tuple(AGENT_TYPES) + tuple(previous) + tuple(p["name"] for p in proposals)
            proposals.extend(propose(llm_client, digest, taken, max_candidates=args.evolve_agents))
    if proposals:
        seen: set[str] = set()
        unique: list[Mapping[str, str]] = []
        for proposal in proposals:
            if proposal["name"] in seen:
                logger.warning("Kandidat %r ignoriert: Name bereits in der Kandidatenliste", proposal["name"])
                continue
            seen.add(proposal["name"])
            unique.append(proposal)
        instances, code_by_name, claim_by_name = prepare_candidates(
            unique,
            instrument=series[0][0] if series else "",
            horizon=args.horizon,
            summary=summary,
        )
        candidates = instances
        candidate_meta = {name: (code_by_name[name], claim_by_name[name]) for name in instances}

    # Batch = die Kandidaten, die Jail/Smoke überstanden haben und jetzt
    # die Gate-Prüfung durchlaufen (exakt das Dict an evaluate_evolved_candidates).
    effective_margin = admission_margin(trials_before + len(candidates), base=ADMISSION_MARGIN)
    artifact = evaluate_evolved_candidates(
        series,
        base_instances,
        candidates,
        candidate_meta,
        previous=previous,
        candle_limit=args.candle_limit,
        min_candles=args.min_candles,
        evaluate_every=args.evaluate_every,
        horizon_bars=args.horizon_bars,
        target_config=target,
        calibration_ratio=args.calibration_ratio,
        max_agents=args.max_evolved,
        promotion_margin=effective_margin,
        summary=summary,
    )
    path = write_json_atomic(agents_path, artifact)
    record_trial_count(trial_path, trials_before + len(candidates))
    write_json_atomic(
        agents_path.with_name(EVOLVED_AGENTS_LAST_RUN_FILENAME),
        {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "trials": trials_before + len(candidates),
            "effective_margin": effective_margin,
            "candidates": summary,
        },
    )
    print(f"Evolved Agents: {len(artifact)} zugelassen ({', '.join(sorted(artifact)) or '—'}) → {path}")
    return 0


def _base_ensemble(args: Namespace) -> dict[str, BaseAgent]:
    """Das 4er-Basis-Ensemble mit den aktuellen Champion-Parametern."""
    from .agent_params import AGENT_TYPES, build_agent, default_params
    from .evolve import load_champion_configs

    config_path = Path(args.configs_output or Path(args.output).with_name("champion_configs.json"))
    previous_configs = load_champion_configs(config_path) or {}
    instances: dict[str, BaseAgent] = {}
    for agent_id in AGENT_TYPES:
        params = default_params(agent_id)
        previous = previous_configs.get(agent_id)
        if previous is not None and isinstance(previous.get("params"), Mapping):
            params = type(params).from_dict(dict(previous["params"]))
        instances[agent_id] = build_agent(agent_id, params)
    return instances


def _default_llm_client(model: str | None) -> LLMClient | None:
    """LLM-Client aus der Umgebung; ``None`` bei Konfig-Fehler (fail-soft)."""
    from packages.llm.client import LLMClient
    from packages.llm.errors import LLMError

    try:
        return LLMClient.from_env(model=model)
    except LLMError as exc:
        logger.warning("LLM nicht konfiguriert (%s: %s) — ohne LLM-Vorschläge", exc.code, exc)
        return None
