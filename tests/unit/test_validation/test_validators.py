"""Tests für die Validation Layer (packages/validation/)."""

from datetime import datetime, timedelta

from packages.validation.validators import (
    CrossFieldValidator,
    DataQualityValidator,
    MarketEventValidator,
    PointInTimeValidator,
)


class TestDataQualityValidator:
    def test_all_pass(self) -> None:
        v = DataQualityValidator()
        results = v.validate(
            probabilities={"up": 0.5, "down": 0.3, "range": 0.2},
            evidence_count=3,
            counter_evidence_count=1,
            data_quality=0.95,
            raw_confidence=0.85,
            calibrated_confidence=0.8,
        )
        passed = [r for r in results if r.passed]
        assert len(passed) == 4  # evidence_minimum + probability_sum + data_quality_score + confidence_calibration

    def test_evidence_minimum_fail(self) -> None:
        v = DataQualityValidator()
        results = v.validate(
            probabilities={"up": 1.0},
            evidence_count=0,
            data_quality=1.0,
        )
        assert results[0].passed is False  # evidence_minimum

    def test_probability_sum_fail(self) -> None:
        v = DataQualityValidator()
        results = v.validate(
            probabilities={"up": 0.6, "down": 0.3},
            evidence_count=1,
            data_quality=1.0,
        )
        assert results[1].passed is False  # probability_sum

    def test_data_quality_below_threshold(self) -> None:
        v = DataQualityValidator()
        results = v.validate(
            probabilities={"up": 0.5, "down": 0.5},
            evidence_count=1,
            data_quality=0.7,
        )
        assert results[2].passed is False  # data_quality_score

    def test_counter_evidence_missing_high_quality(self) -> None:
        v = DataQualityValidator()
        results = v.validate(
            probabilities={"up": 0.5, "down": 0.5},
            evidence_count=3,
            counter_evidence_count=0,
            data_quality=0.95,
        )
        assert results[3].passed is False  # counter_evidence_missing

    def test_confidence_calibrated_fail(self) -> None:
        v = DataQualityValidator()
        results = v.validate(
            probabilities={"up": 0.5, "down": 0.5},
            evidence_count=1,
            raw_confidence=0.7,
            calibrated_confidence=0.9,
        )
        assert results[3].passed is False  # confidence_calibration


class TestPointInTimeValidator:
    def test_all_pass(self) -> None:
        v = PointInTimeValidator()
        now = datetime(2024, 1, 1, 12, 0, 0)
        results = v.validate(
            analysis_time=now,
            availability_time=now - timedelta(hours=1),
            ingestion_time=now - timedelta(hours=2),
            event_time=now - timedelta(hours=3),
            source_quality=0.9,
        )
        assert all(r.passed for r in results)

    def test_lookahead_detected(self) -> None:
        v = PointInTimeValidator()
        now = datetime(2024, 1, 1, 12, 0, 0)
        results = v.validate(
            analysis_time=now,
            availability_time=now + timedelta(hours=1),  # future!
            ingestion_time=now - timedelta(hours=2),
            event_time=now - timedelta(hours=3),
            source_quality=0.9,
        )
        assert results[0].passed is False  # no_lookahead_availability

    def test_event_in_future(self) -> None:
        v = PointInTimeValidator()
        now = datetime(2024, 1, 1, 12, 0, 0)
        results = v.validate(
            analysis_time=now,
            availability_time=now - timedelta(hours=1),
            ingestion_time=now - timedelta(hours=2),
            event_time=now + timedelta(hours=1),  # future event
            source_quality=0.9,
        )
        assert results[2].passed is False  # event_not_future

    def test_ingestion_before_event(self) -> None:
        v = PointInTimeValidator()
        now = datetime(2024, 1, 1, 12, 0, 0)
        results = v.validate(
            analysis_time=now,
            availability_time=now - timedelta(hours=1),
            ingestion_time=now - timedelta(hours=4),  # before event!
            event_time=now - timedelta(hours=3),
            source_quality=0.9,
        )
        assert results[1].passed is False  # ingestion_after_event

    def test_source_quality_below_threshold(self) -> None:
        v = PointInTimeValidator()
        now = datetime(2024, 1, 1, 12, 0, 0)
        results = v.validate(
            analysis_time=now,
            availability_time=now - timedelta(hours=1),
            ingestion_time=now - timedelta(hours=2),
            event_time=now - timedelta(hours=3),
            source_quality=0.3,
        )
        assert results[3].passed is False  # source_quality_threshold


class TestCrossFieldValidator:
    def test_all_pass(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            risk_approved=True,
            has_hard_block=False,
            decision_type="LONG_BIAS",
            reason="Strong momentum signal",
            max_position_size=0.5,
            reduction_factor=1.0,
            portfolio_exposure_ratio=0.3,
            portfolio_max_exposure=1.0,
            is_long=True,
            is_short=False,
            is_range=False,
        )
        assert all(r.passed for r in results)

    def test_hard_block_with_approved(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            risk_approved=True,
            has_hard_block=True,
            decision_type="LONG_BIAS",
            reason="Test",
            blocking_reasons=["exposure"],
            max_position_size=0.0,
        )
        assert results[1].passed is False  # veto_consistency

    def test_no_trade_missing_reason(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            decision_type="NO_TRADE_DATA_QUALITY",
            reason="",
            blocking_reasons=["low_quality"],
            is_long=False,
            is_short=False,
        )
        reason_result = next(r for r in results if r.check == "no_trade_reason")
        assert reason_result.passed is False

    def test_hard_block_veto_overrides(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            has_hard_block=True,
            decision_type="LONG_BIAS",
            reason="Test",
            blocking_reasons=["data_quality"],
            is_long=False,
        )
        veto_result = next(r for r in results if r.check == "veto_overrides_decision")
        assert veto_result.passed is False

    def test_portfolio_exposure_exceeded(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            decision_type="LONG_BIAS",
            reason="Test",
            blocking_reasons=[],
            portfolio_exposure_ratio=1.5,
            portfolio_max_exposure=1.0,
            is_long=False,
        )
        exposure_result = next(r for r in results if r.check == "portfolio_exposure_limit")
        assert exposure_result.passed is False

    def test_max_position_size_out_of_range(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            decision_type="LONG_BIAS",
            reason="Test",
            max_position_size=1.5,
            is_long=False,
        )
        assert results[0].passed is False  # max_position_size_range

    def test_no_trade_missing_blocking_reasons(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            decision_type="NO_TRADE_RISK",
            reason="Risk gate failed",
            blocking_reasons=[],
            is_long=False,
        )
        blocking_result = next(r for r in results if r.check == "no_trade_blocking_reasons")
        assert blocking_result.passed is False

    def test_reduction_factor_out_of_range(self) -> None:
        v = CrossFieldValidator()
        results = v.validate(
            decision_type="LONG_BIAS",
            reason="Test",
            reduction_factor=1.5,
            is_long=False,
        )
        rf_result = next(r for r in results if r.check == "reduction_factor_range")
        assert rf_result.passed is False


class TestMarketEventValidator:
    def test_candle_valid(self) -> None:
        v = MarketEventValidator()
        results = v.validate_candle(open_price=100, high_price=110, low_price=95, close_price=105)
        assert all(r.passed for r in results)

    def test_candle_low_zero(self) -> None:
        v = MarketEventValidator()
        results = v.validate_candle(open_price=100, high_price=110, low_price=0, close_price=105)
        assert results[0].passed is False  # candle_low_positive

    def test_candle_high_below_low(self) -> None:
        v = MarketEventValidator()
        results = v.validate_candle(open_price=100, high_price=90, low_price=95, close_price=105)
        assert results[1].passed is False  # candle_high_gte_low

    def test_trade_valid(self) -> None:
        v = MarketEventValidator()
        results = v.validate_trade(price=50000, quantity=1.5)
        assert all(r.passed for r in results)

    def test_trade_zero_price(self) -> None:
        v = MarketEventValidator()
        results = v.validate_trade(price=0, quantity=1.5)
        assert results[0].passed is False  # trade_price_positive

    def test_orderbook_valid(self) -> None:
        v = MarketEventValidator()
        results = v.validate_orderbook(bid_count=100, ask_count=80)
        assert all(r.passed for r in results)

    def test_orderbook_empty_asks(self) -> None:
        v = MarketEventValidator()
        results = v.validate_orderbook(bid_count=100, ask_count=0)
        assert results[1].passed is False  # orderbook_asks

    def test_dispatch_candle(self) -> None:
        v = MarketEventValidator()
        results = v.validate(event_type="candle", open=100, high=110, low=95, close=105)
        assert all(r.passed for r in results)

    def test_dispatch_trade(self) -> None:
        v = MarketEventValidator()
        results = v.validate(event_type="trade", price=50000, quantity=1.5)
        assert all(r.passed for r in results)

    def test_dispatch_orderbook(self) -> None:
        v = MarketEventValidator()
        results = v.validate(event_type="orderbook", bid_count=100, ask_count=80)
        assert all(r.passed for r in results)

    def test_dispatch_unknown(self) -> None:
        v = MarketEventValidator()
        results = v.validate(event_type="foobar")
        assert results[0].passed is False
