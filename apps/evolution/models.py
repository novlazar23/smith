"""Preregistrierungsschema der Evolutions-Pipeline.

Kern des Data-Snooping-Schutzes: Eine Hypothese ist VOR dem Test vollständig
fixiert (Claim, Variante, Testplan, Entscheidungsregel, Quelle). Änderungen
nach dem Test = neue Hypothese (neue ID) — keine stillschweigende
Nachbesserung.

Zwei Arten (``kind``):

- ``config``: Parameter-Variante einer registrierten Strategie,
- ``mechanism``: neue Regel-Strategie (Code aus dem Mechanismus-Pfad;
  ``Variant.code`` trägt den Quelltext, ``code_file`` den Zielpfad).

Die Entscheidungsregel ist rein mechanisch (kein LLM): OOS-Marge über der
Baseline, Mindest-Legs, Drawdown-Cap, Mehrheiten-Bedingung — plus
Multi-Testing-Deflation über die Familien-Zählung (s. ``evaluate.judge``).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

#: Muster für Strategie-/Mechanismus-Namen (klein, underscore, 3-40 Zeichen).
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def utcnow_iso() -> str:
    """Aktuelle UTC-Zeit als ISO-String (Sekundenpräzision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_hypothesis_id(family: str, existing_ids: set[str]) -> str:
    """Eindeutige Hypothesen-ID: ``<family>-YYYYMMDD-##`` (nächster freier Index)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    prefix = f"{family}-{stamp}-"
    taken = {int(i.rsplit("-", 1)[1]) for i in existing_ids if i.startswith(prefix) and i.rsplit("-", 1)[1].isdigit()}
    seq = max(taken, default=0) + 1
    return f"{prefix}{seq}"


class TestPlan(BaseModel):
    """Fixierter Testplan: Fenster, Assets, Auflösung, Mindestkerzen.

    ``oos_end=None`` → letzte verfügbare Kerze. Fenster sind diskret
    (Kalibrierung strikt vor OOS) — keine Überlappung.
    """

    instruments: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    timeframe: str = "5m"
    calibration_start: str = "2021-05-01"
    calibration_end: str = "2022-12-31"
    oos_start: str = "2023-01-01"
    oos_end: str | None = None
    #: Fenster mit weniger Kerzen werden übersprungen (Data-Gap, nicht fatal).
    min_candles: int = Field(default=300, ge=10)

    @model_validator(mode="after")
    def _check_windows(self) -> TestPlan:
        if not self.instruments:
            raise ValueError("instruments darf nicht leer sein")
        if self.calibration_start >= self.calibration_end:
            raise ValueError("Kalibrierungsfenster ungültig (start >= end)")
        if self.oos_start <= self.calibration_end:
            raise ValueError("OOS-Fenster überlappt das Kalibrierungsfenster (OOS muss nach dem Kalibrierungsende beginnen)")
        return self


class DecisionRule(BaseModel):
    """Mechanische Promotions-Regel (kein LLM).

    Alle Bedingungen müssen gleichzeitig gelten, sonst Rejection.
    ``oos_margin_min_pct`` wird bei ``n_family >= 10`` im Judge verdoppelt
    (grobe Multi-Testing-Deflation, dokumentiert im Digest).
    """

    #: OOS-Portfolio-Mittel (Ø Asset-Rendite) ≥ Baseline + Marge (pp).
    oos_margin_min_pct: float = Field(default=1.0, ge=0.0)
    #: Mindestanzahl Round-Trip-Legs über alle OOS-Asset-Fenster.
    min_legs: int = Field(default=10, ge=1)
    #: Maximaler Drawdown pro OOS-Asset-Fenster (pp).
    max_dd_pct: float = Field(default=8.0, gt=0.0)
    #: ≥ 50 % der OOS-Asset-Fenster müssen positiv sein.
    majority_positive: bool = True


class Variant(BaseModel):
    """Konkrete Variante: Strategie + Parameter (Konfig) bzw. + Code (Mechanismus)."""

    strategy: str
    params: dict[str, float] = Field(default_factory=dict)
    #: Mechanismus: Quelltext der ``RuleStrategy``-Subklasse (Jail-geprüft).
    code: str | None = None
    #: Mechanismus: Zielpfad der Strategie-Datei (``packages/strategies/<name>.py``).
    code_file: str | None = None

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> Variant:
        if not NAME_PATTERN.match(self.strategy):
            raise ValueError(f"Strategie-Name {self.strategy!r} verletzt {NAME_PATTERN.pattern}")
        if (self.code is None) != (self.code_file is None):
            raise ValueError("code und code_file müssen zusammen gesetzt sein (Mechanismus)")
        return self


class Hypothesis(BaseModel):
    """Preregistrierte Hypothese (unveränderlich nach Fixierung, außer Status/Feldern)."""

    id: str
    created_at: str
    family: str
    kind: Literal["config", "mechanism"]
    claim: str
    variant: Variant
    test_plan: TestPlan = Field(default_factory=TestPlan)
    decision_rule: DecisionRule = Field(default_factory=DecisionRule)
    #: Herkunft: ``manual``, ``sweep``, ``llm:<persona>``, ``human``.
    source: str = "manual"
    status: Literal["proposed", "testing", "passed", "rejected", "pending", "error"] = "proposed"
    #: Ergebnisse der Test-Matrix (gefüllt nach dem Test; Schema s. ``evaluate``).
    results: dict[str, Any] | None = None
    #: Verdict des Judges (``{"decision", "reasons", ...}``; gefüllt nach dem Test).
    verdict: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_required(self) -> Hypothesis:
        if not NAME_PATTERN.match(self.family):
            raise ValueError(f"Family-Name {self.family!r} verletzt {NAME_PATTERN.pattern}")
        if len(self.claim.strip()) < 10:
            raise ValueError("claim zu kurz (mind. 10 Zeichen — ein belastbarer Satz)")
        if self.kind == "mechanism" and self.variant.code is None:
            raise ValueError("mechanism-Hypothese benötigt variant.code")
        return self


class Proposal(BaseModel):
    """Vorschlag einer Quelle (LLM-Personas, Grid-Sweep, manuell).

    Wird vom Zyklus zu einer `Hypothesis` (Preregistrierung) — die
    Pflichtfelder (id/created_at) werden beim Preregistrieren gesetzt.
    """

    family: str
    kind: Literal["config", "mechanism"]
    claim: str
    variant: Variant
    test_plan: TestPlan | None = None
    decision_rule: DecisionRule | None = None

    @model_validator(mode="after")
    def _check_family_matches(self) -> Proposal:
        if self.family != self.variant.strategy:
            raise ValueError("family muss mit variant.strategy übereinstimmen")
        if self.kind == "mechanism" and not self.variant.code:
            raise ValueError("mechanism-Proposal benötigt variant.code")
        return self


def hypothesis_from_proposal(
    proposal: Proposal,
    existing_ids: set[str],
    source: str,
) -> Hypothesis:
    """Erzeugt die Preregistrierung aus einem Vorschlag (ID + Zeitstempel)."""
    return Hypothesis(
        id=make_hypothesis_id(proposal.family, existing_ids),
        created_at=utcnow_iso(),
        family=proposal.family,
        kind=proposal.kind,
        claim=proposal.claim.strip(),
        variant=proposal.variant,
        test_plan=proposal.test_plan or TestPlan(),
        decision_rule=proposal.decision_rule or DecisionRule(),
        source=source,
    )


def variant_key(variant: Variant) -> str:
    """Stable Deduplication-Schlüssel: Strategie + sortierte Parameter (ohne Code)."""
    params = ",".join(f"{k}={v:g}" for k, v in sorted(variant.params.items()))
    return f"{variant.strategy}::{params}"
