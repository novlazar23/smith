"""Zyklus-Orchestrierung: Diskutieren → Preregistrieren → Testen → Beurteilen.

Ein Zyklus ist budgetiert (``CycleBudget``): maximal N neue Vorschläge,
maximal M Engine-Runs, maximal W Minuten Wanduhr. Übrigbleibende
Hypothesen bleiben ``pending`` und werden im nächsten Zyklus fortgesetzt
(der State ist die einzige Quelle der Wahrheit — der Zyklus ist
idempotenz-tauglich: bereits getestete Varianten werden nie retestet).
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from packages.llm.client import LLMClient

from .digest import build_digest, fetch_live_paper, write_digest
from .evaluate import FeedFactory, judge, run_variant_matrix
from .mechanism import (
    apply_registry,
    revert_registry,
    run_clone_check,
    sample_clone_candles,
    smoke_test,
    validate_mechanism_code,
    write_strategy_file,
)
from .models import Hypothesis, Proposal, hypothesis_from_proposal
from .state import EvolutionStore

logger = logging.getLogger(__name__)


@dataclass
class CycleBudget:
    """Zyklus-Budget (Varianten, Engine-Runs, Wanduhr)."""

    max_proposals: int = 3
    max_runs: int = 40
    max_minutes: float = 120.0
    _start: float = field(default_factory=time.monotonic, init=False)
    _runs: int = field(default=0, init=False)

    def spend_runs(self, n: int = 1) -> None:
        self._runs += n

    @property
    def runs_left(self) -> int:
        return self.max_runs - self._runs

    def exhausted(self) -> bool:
        if self._runs >= self.max_runs:
            return True
        elapsed_minutes = (time.monotonic() - self._start) / 60.0
        return elapsed_minutes >= self.max_minutes


def _repo_root() -> Path:
    return Path.cwd()


def _prepare_mechanism(
    hypothesis: Hypothesis,
    feed_factory: FeedFactory,
) -> str | None:
    """Mechanismus-Vorbereitung: Jail → Smoke → Code+Registry → Clone-Check.

    Returns Ablehnungs-Grund oder ``None`` (Erfolg). Bei jedem Fehler wird
    der teilweise angesprochene Zustand (Datei, Registry) rückgängig gemacht.
    """
    name = hypothesis.variant.strategy
    code = hypothesis.variant.code or ""
    repo = _repo_root()

    problem = validate_mechanism_code(code, name)
    if problem:
        return f"Jail: {problem}"
    problem = smoke_test(code, name)
    if problem:
        return f"Smoke-Test: {problem}"

    write_strategy_file(code, name, repo)
    try:
        apply_registry(code, name, repo)
    except ValueError as exc:
        _remove_strategy_file(name, repo)
        return f"Registry: {exc}"

    # Clone-Check-Periode: letzte 90 Tage 5m-Kerzen des ersten Plan-Assets.
    instrument = hypothesis.test_plan.instruments[0]
    start = (datetime.now(UTC) - timedelta(days=90)).isoformat(timespec="seconds")
    try:
        candles = feed_factory(instrument, start, None, "5m")
    except Exception as exc:
        _revert_mechanism(code, name, repo)
        return f"Clone-Check: Kerzen nicht ladbar ({type(exc).__name__}: {exc})"
    if candles:
        clone = run_clone_check(name, dict(hypothesis.variant.params), sample_clone_candles(candles))
        if clone is not None:
            _revert_mechanism(code, name, repo)
            return f"Clone von {clone!r} (Overlap ≥ 90 % — Konfig-Variante, kein neuer Mechanismus)"
    logger.info("Mechanismus %s vorbereitet (Jail+Smoke+Registry+Clone-Check)", name)
    return None


def _revert_mechanism(code: str, name: str, repo: Path) -> None:
    revert_registry(code, name, repo)
    _remove_strategy_file(name, repo)


def _remove_strategy_file(name: str, repo: Path) -> None:
    path = repo / "packages" / "strategies" / f"{name}.py"
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _graveyard_entry(hypothesis: Hypothesis, reasons: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from .models import variant_key

    entry: dict[str, Any] = {
        "id": hypothesis.id,
        "family": hypothesis.family,
        "kind": hypothesis.kind,
        "variant_key": variant_key(hypothesis.variant),
        "claim": hypothesis.claim,
        "reasons": reasons,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)
    return entry


def _test_one(
    store: EvolutionStore,
    hypothesis: Hypothesis,
    feed_factory: FeedFactory,
    budget: CycleBudget,
) -> str:
    """Testet eine einzelne Hypothese (Mechanismus-Vorbereitung + Matrix + Judge)."""
    hypothesis.status = "testing"
    store.update_hypothesis(hypothesis)

    if hypothesis.kind == "mechanism":
        problem = _prepare_mechanism(hypothesis, feed_factory)
        if problem is not None:
            hypothesis.status = "error"
            hypothesis.verdict = {"decision": "rejected", "reasons": [problem]}
            store.update_hypothesis(hypothesis)
            store.add_graveyard(_graveyard_entry(hypothesis, [problem]))
            logger.warning("Hypothese %s verworfen: %s", hypothesis.id, problem)
            return "error"

    matrix, error = run_variant_matrix(hypothesis, feed_factory, store.baseline())
    if matrix is None:
        hypothesis.status = "error"
        hypothesis.verdict = {"decision": "rejected", "reasons": [error or "Test-Matrix fehlgeschlagen"]}
        store.update_hypothesis(hypothesis)
        store.add_graveyard(_graveyard_entry(hypothesis, [error or "Test-Matrix fehlgeschlagen"]))
        return "error"

    # Runs gezählt: 2 Configs (Variante + Baseline) x Assets pro Fenster.
    n_runs = sum(2 * len(window_assets) for window_assets in matrix["windows"].values())
    budget.spend_runs(n_runs)

    verdict = judge(hypothesis, matrix, store.family_count(hypothesis.family))
    hypothesis.results = matrix
    hypothesis.verdict = verdict

    oos = matrix["portfolio"].get("oos", {})
    metrics_summary = {
        "oos_delta_pp": verdict.get("delta_pp"),
        "oos_variant_mean_pct": oos.get("variant_mean_pct"),
        "oos_baseline_mean_pct": oos.get("baseline_mean_pct"),
        "legs": oos.get("legs"),
        "max_dd_pct": oos.get("max_dd_max_pct"),
        "bootstrap": matrix.get("bootstrap"),
    }

    if verdict["decision"] == "promoted":
        hypothesis.status = "passed"
        store.update_hypothesis(hypothesis)
        store.promote(
            {
                "id": hypothesis.id,
                "family": hypothesis.family,
                "kind": hypothesis.kind,
                "claim": hypothesis.claim,
                "variant": hypothesis.variant.model_dump(exclude_none=True),
                "metrics": metrics_summary,
                "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        logger.info("Hypothese %s PROMOTED (%s)", hypothesis.id, metrics_summary["oos_delta_pp"])
        return "promoted"

    hypothesis.status = "rejected"
    store.update_hypothesis(hypothesis)
    store.add_graveyard(_graveyard_entry(hypothesis, verdict["reasons"], metrics_summary))
    logger.info("Hypothese %s rejected: %s", hypothesis.id, "; ".join(verdict["reasons"]))
    return "rejected"


def run_cycle(
    store: EvolutionStore,
    feed_factory: FeedFactory,
    *,
    with_llm: bool = False,
    llm_client: LLMClient | None = None,
    model: str | None = None,
    proposals: list[Proposal] | None = None,
    budget: CycleBudget | None = None,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Führt einen Evolutions-Zyklus aus. Returns Zusammenfassung (dict).

    Args:
        store: Evolutions-State.
        feed_factory: Kerzen-Lader (CH im Betrieb, synthetisch in Tests).
        with_llm: Personas-Aufruf aktiviert (braucht LLM-Umgebung).
        llm_client: Injizierter Client (Tests); sonst ``LLMClient.from_env``.
        model: Modell-Override für die Personas.
        proposals: Externe Vorschläge (Tests/``--propose``); ergänzen die
            LLM-Vorschläge (Budget gilt für beide zusammen).
        budget: Zyklus-Budget (Default: 3 Vorschläge, 40 Runs, 120 min).
        export_dir: Optionaler Export-Zielordner (State nach dem Zyklus).
    """
    budget = budget or CycleBudget()
    counts = {"promoted": 0, "rejected": 0, "pending": 0, "error": 0, "tested": 0}
    verdicts: list[dict[str, Any]] = []

    digest = build_digest(store, live=fetch_live_paper())
    logger.info("Zyklus gestartet (LLM=%s, Budget: %d Vorschläge/%d Runs/%.0f min)", with_llm, budget.max_proposals, budget.max_runs, budget.max_minutes)

    # 1) Diskutieren (LLM) + externe Vorschläge → Preregistrierung
    collected: list[tuple[Proposal, str]] = [(p, "external") for p in (proposals or [])]
    if with_llm and len(collected) < budget.max_proposals:
        from .personas import propose

        if llm_client is None:
            from packages.llm.client import LLMClient

            llm_client = LLMClient.from_env(model=model)
        for proposal in propose(llm_client, digest, max_proposals=budget.max_proposals - len(collected)):
            collected.append((proposal, "llm:personas"))

    existing_ids = store.hypothesis_ids()
    n_new = 0
    for proposal, source in collected[: budget.max_proposals]:
        if store.find_duplicate(proposal.variant) is not None:
            logger.info("Vorschlag übersprungen (Varianten-Duplikat): %s", proposal.variant.strategy)
            continue
        if store.in_graveyard(proposal.variant) is not None:
            logger.info("Vorschlag übersprungen (liegt im Grab): %s", proposal.variant.strategy)
            continue
        hypothesis = hypothesis_from_proposal(proposal, existing_ids, source)
        store.add_hypothesis(hypothesis)
        existing_ids.add(hypothesis.id)
        n_new += 1

    # 2) Testen + Beurteilen (alle offenen Hypothesen, Budget-geprüft)
    queue = [h for h in store.all_hypotheses() if h.status in ("proposed", "pending")]
    for hypothesis in queue:
        if budget.exhausted():
            hypothesis.status = "pending"
            store.update_hypothesis(hypothesis)
            counts["pending"] += 1
            continue
        outcome = _test_one(store, hypothesis, feed_factory, budget)
        counts[outcome] += 1
        counts["tested"] += 1
        oos = (hypothesis.results or {}).get("portfolio", {}).get("oos", {})
        verdicts.append(
            {
                "id": hypothesis.id,
                "decision": (hypothesis.verdict or {}).get("decision"),
                "delta_pp": (hypothesis.verdict or {}).get("delta_pp"),
                "oos_variant_mean_pct": oos.get("variant_mean_pct"),
            }
        )

    # 3) State aktualisieren + Digest protokollieren
    store.set_meta(
        last_cycle={
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_proposals_new": n_new,
            "n_tested": counts["tested"],
            "n_promoted": counts["promoted"],
            "n_rejected": counts["rejected"],
            "n_pending": counts["pending"],
            "n_error": counts["error"],
            "verdicts": verdicts,
        }
    )
    final_digest = build_digest(store, live=fetch_live_paper())
    write_digest(store, final_digest)
    if export_dir is not None:
        from .export import export_state

        export_state(store, export_dir)

    summary = {
        "new_proposals": n_new,
        **counts,
        "last_cycle": store.meta().get("last_cycle"),
    }
    logger.info("Zyklus fertig: %s", summary)
    return summary
