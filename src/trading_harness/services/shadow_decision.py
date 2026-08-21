"""Deterministische Shadow-Trading-Decision-Funktionen (WI-ST-04, Spec ST.6).

Dieses Modul ist der MVP-Ersatz für ein Consensus-Modul (Spec Z5: es existiert
keins im Codebase) und enthält genau drei pure Funktionen:

- ``aggregate_signals`` — Mehrheit der NON-``NO_TRADE``-Signale bestimmt die
  Richtung (LONG/SHORT); Gleichstand oder alle ``NO_TRADE`` -> ``NO_TRADE``;
  Confidence-Gate über ``min_confidence`` (Spec ``shadow_min_confidence``,
  Default 0.6).
- ``no_trade_reason`` — mappt eine ``NO_TRADE``-Aggregation auf genau eines
  der zwei Gründe: ``NO_ACTIVE_CHAMPIONS`` (signal_count == 0) oder
  ``BELOW_MIN_CONFIDENCE`` (alle übrigen NO_TRADE-Fälle: Tie und
  Confidence-gate). Diese zwei Strings sind das komplette
  NO_TRADE-Gründervokabular (Epic WI-ST-04 / E2); der Loop speichert sie in
  ``ShadowTradingRecord.risk_reason``.
- ``build_trade_proposal`` — deterministisches ``TradeProposal`` für
  freigegebene Trades.

Z4-Schnitt: Der Agenten-Statusfilter (``ACTIVE``/``CHAMPION``) passiert im
ShadowTradingLoop (WI-ST-05), NICHT in diesem Modul: ``AgentSignal`` trägt
kein Agenten-Status-Feld. Die Loop sammelt nur Signale qualifizierender
Agenten; eine leere Signal-Liste bedeutet daher "keine qualifizierenden
Agenten" und wird hier zu ``NO_TRADE`` mit Grund ``NO_ACTIVE_CHAMPIONS``.

Purity-Garantien: keine Funktionen dieses Moduls führen I/O aus, nutzen
Zufallszahlen, die Uhr, die Konfiguration (``get_settings``) oder LLM-Ausgaben
zur Wertebildung. Alle Parameter sind explizit — die Loop übergibt die
Config-Werte (``shadow_min_confidence``, ``shadow_stop_loss_fraction``,
``shadow_min_risk_reward``) als Argumente. Determinismus: gleiche Eingaben
ergeben bit-identische Ergebnisse (Spec ST.6, Akzeptanzkriterium e).
"""

from __future__ import annotations

from trading_harness.models import (
    AgentSignal,
    MarketSnapshot,
    PortfolioState,
    RiskDecision,
    SignalAggregation,
    TradeProposal,
)

#: Komplettes NO_TRADE-Gründervokabular (Epic WI-ST-04 / E2).
NO_ACTIVE_CHAMPIONS = "NO_ACTIVE_CHAMPIONS"
BELOW_MIN_CONFIDENCE = "BELOW_MIN_CONFIDENCE"

_DIRECTION_LONG = "LONG"
_DIRECTION_SHORT = "SHORT"
_DIRECTION_NO_TRADE = "NO_TRADE"


def aggregate_signals(
    signals: list[AgentSignal],
    symbol: str,
    min_confidence: float = 0.6,
) -> SignalAggregation:
    """Aggregiert Agenten-Signale eines Symbols deterministisch (Spec ST.6).

    Args:
        signals: ``AgentSignal``-Objekte, die die Loop für ``symbol`` von
            Agenten mit Status ``ACTIVE`` oder ``CHAMPION`` gesammelt hat
            (Z4). Der Statusfilter selbst liegt bei der Loop; eine leere
            Liste bedeutet "keine qualifizierenden Agenten".
        symbol: Symbol, auf das sich die Signale beziehen (API-Kontext; das
            Ergebnis-Modell ``SignalAggregation`` speichert es nicht).
        min_confidence: Confidence-Gate, Spec ``shadow_min_confidence``
            (Default 0.6). Als Parameter exponiert, damit die Funktion pure
            bleibt; die Loop übergibt ``get_settings().shadow_min_confidence``.

    Semantik:
        * ``signal_count`` zählt Signale mit Richtung LONG/SHORT;
          ``no_trade_count`` zählt die übrigen (NO_TRADE)-Signale.
        * Die Mehrheit der NON-NO_TRADE-Signale entscheidet: mehr LONG als
          SHORT -> ``"LONG"``, mehr SHORT als LONG -> ``"SHORT"``;
          Gleichstand (inklusive 0 zu 0) -> ``"NO_TRADE"``.
        * ``confidence`` ist das arithmetische Mittel der Confidences der
          Signale der gewählten Richtung. Bei NO_TRADE durch Gleichstand ist
          ``confidence`` 0.0. Wurde eine Richtung gewählt, deren mittleres
          Confidence streng unter ``min_confidence`` liegt (genau der
          Schwellwert besteht), wird die Richtung zu ``"NO_TRADE"``
          herabgestuft und der berechnete Mittelwert bleibt als Information
          in ``confidence`` erhalten.
        * ``agent_ids`` enthält die ``agent_id`` aller Eingabesignale in
          Eingabe-Reihenfolge (Audit: alle Teilnehmer, nicht nur die
          Mehrheit).

    Pure: kein I/O, kein Zufall, keine Uhr. Deterministisch: gleiche Eingaben
    liefern ein bit-identisches ``SignalAggregation`` (Spec ST.6, e).
    """
    long_signals = [s for s in signals if s.direction == _DIRECTION_LONG]
    short_signals = [s for s in signals if s.direction == _DIRECTION_SHORT]
    signal_count = len(long_signals) + len(short_signals)
    no_trade_count = len(signals) - signal_count

    direction = _DIRECTION_NO_TRADE
    confidence = 0.0
    if len(long_signals) > len(short_signals):
        direction = _DIRECTION_LONG
        confidence = sum(s.confidence for s in long_signals) / len(long_signals)
    elif len(short_signals) > len(long_signals):
        direction = _DIRECTION_SHORT
        confidence = sum(s.confidence for s in short_signals) / len(short_signals)

    if direction != _DIRECTION_NO_TRADE and confidence < min_confidence:
        direction = _DIRECTION_NO_TRADE

    return SignalAggregation(
        direction=direction,
        confidence=confidence,
        signal_count=signal_count,
        no_trade_count=no_trade_count,
        agent_ids=[s.agent_id for s in signals],
    )


def no_trade_reason(aggregation: SignalAggregation) -> str | None:
    """Liefert den deterministischen NO_TRADE-Grund einer Aggregation (E2).

    Returns:
        * ``None``, wenn ``aggregation.direction`` nicht ``"NO_TRADE"`` ist.
        * ``"NO_ACTIVE_CHAMPIONS"``, wenn ``signal_count == 0``: es gab
          keine Signale, d. h. die Loop fand keine ``ACTIVE``/``CHAMPION``-
          Agenten (Z4).
        * ``"BELOW_MIN_CONFIDENCE"`` in jedem anderen NO_TRADE-Fall — deckt
          sowohl LONG/SHORT-Gleichstand als auch eine Richtung ab, deren
          mittleres Confidence unter dem ``shadow_min_confidence``-Gate
          lag. Diese beiden Strings sind das vollständige
          NO_TRADE-Gründervokabular; der Loop speichert den Wert in
          ``ShadowTradingRecord.risk_reason``.
    """
    if aggregation.direction != _DIRECTION_NO_TRADE:
        return None
    if aggregation.signal_count == 0:
        return NO_ACTIVE_CHAMPIONS
    return BELOW_MIN_CONFIDENCE


def build_trade_proposal(
    aggregation: SignalAggregation,
    snapshot: MarketSnapshot,
    portfolio_state: PortfolioState,
    risk_decision: RiskDecision,
    stop_loss_fraction: float = 0.02,
    min_risk_reward: float = 2.0,
) -> TradeProposal:
    """Baut das deterministische ``TradeProposal`` für einen freigegebenen Trade.

    Nur für freigegebene Trades aufrufen, deren Aggregation Richtung
    ``"LONG"`` oder ``"SHORT"`` hat: die Loop läuft zuvor die deterministische
    RiskEngine und ruft diese Funktion nur bei ``approved``.

    Preisformeln (auschließlich aus den Config-Fraktionen
    ``stop_loss_fraction`` / ``min_risk_reward`` gesteuert, die die Loop aus
    ``get_settings()`` übergibt — nur lesbar, von keiner
    ``AgentSignal``/LLM-Ausgabe ableit- oder überschreibbar):

        stop_distance  = entry_price * stop_loss_fraction
        LONG:           stop_price   = entry_price - stop_distance
                         target_price = entry_price + stop_distance * min_risk_reward
        SHORT:          stop_price   = entry_price + stop_distance
                         target_price = entry_price - stop_distance * min_risk_reward

    Mengenformel (gedeckelt durch die deterministische Risiko-Entscheidung):

        requested_quantity = (equity * risk_decision.risk_fraction) / stop_distance
        requested_quantity = min(requested_quantity,
                                 risk_decision.max_position_size)

    Ticker-Vertrag (Spec ST.1): Der Entry-Preis wird vom festen Schlüssel
    ``"ticker"`` in ``snapshot.data`` gelesen; die Loop speichert den
    geholten Preis dort. Fehlender, nicht-numerischer oder nicht-positiver
    Ticker wirft ``ValueError`` (die Loop mappt das auf SKIPPED_MARKET_DATA).

    ``decision_id`` wird ausschließlich deterministisch aus den Eingaben
    abgeleitet — kein uuid, keine Uhr:
    ``f"shadow-dec-{snapshot.id}-{snapshot.symbol}"`` — identische
    Loop-Durchläufe erzeugen damit identische decision_ids (Spec ST.14) und
    wiederholte Entscheidungen auf demselben Snapshot sind deduplizierbar.

    ``open_positions`` ist ``len(portfolio_state.positions)``;
    ``requested_leverage`` bleibt der Modell-Default 1.0, und
    ``current_daily_loss_fraction`` / ``current_portfolio_risk_fraction`` /
    ``expected_slippage_bps`` bleiben die Modell-Defaults 0.0 (die Loop
    befüllt sie in einem späteren Arbeitspaket, falls/sobald nötig).

    Raises:
        ValueError: Aggregations-Richtung ist nicht LONG/SHORT, Ticker
            ungültig (fehlend, nicht-numerisch, <= 0), Equity <= 0,
            stop_distance <= 0 oder die gedeckelte requested_quantity <= 0.
    """
    if aggregation.direction not in (_DIRECTION_LONG, _DIRECTION_SHORT):
        raise ValueError(
            "build_trade_proposal requires direction LONG or SHORT, "
            f"got {aggregation.direction!r}"
        )

    try:
        entry_price = float(snapshot.data["ticker"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("snapshot.data must contain a numeric 'ticker' price > 0") from exc
    if entry_price <= 0:
        raise ValueError("snapshot.data['ticker'] must be > 0")

    stop_distance = entry_price * stop_loss_fraction
    if stop_distance <= 0:
        raise ValueError("stop distance must be > 0 (stop_loss_fraction must be > 0)")

    equity = portfolio_state.current_equity
    if equity <= 0:
        raise ValueError("portfolio_state.current_equity must be > 0")

    if aggregation.direction == _DIRECTION_LONG:
        side = "BUY"
        stop_price = entry_price - stop_distance
        target_price = entry_price + stop_distance * min_risk_reward
    else:
        side = "SELL"
        stop_price = entry_price + stop_distance
        target_price = entry_price - stop_distance * min_risk_reward

    requested_quantity = (equity * risk_decision.risk_fraction) / stop_distance
    requested_quantity = min(requested_quantity, risk_decision.max_position_size)
    if requested_quantity <= 0:
        raise ValueError(
            "requested_quantity must be > 0 (check risk_fraction and max_position_size)"
        )

    return TradeProposal(
        decision_id=f"shadow-dec-{snapshot.id}-{snapshot.symbol}",
        symbol=snapshot.symbol,
        side=side,
        equity=equity,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        open_positions=len(portfolio_state.positions),
        requested_quantity=requested_quantity,
    )
