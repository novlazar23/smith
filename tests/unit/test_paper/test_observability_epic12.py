"""Tests for EPIC-12 observability package metrics."""

from __future__ import annotations

from packages.observability.metrics import MetricsRegistry


class TestEpic12AgentMetrics:
    """Test EPIC-12 agent run/failure/duration/schema error metrics."""

    def test_agent_runs_total(self) -> None:
        reg = MetricsRegistry(namespace="epic12_agent")
        reg.record_agent_run("trend_follower", "success", duration=0.5)
        reg.record_agent_run("trend_follower", "failure", duration=0.1)
        reg.record_agent_run("sentiment_agent", "success", duration=1.2)
        text = reg.get_metrics_text().decode()
        assert "epic12_agent_agent_runs_total{agent_id=\"trend_follower\"" in text
        assert "epic12_agent_agent_runs_total{agent_id=\"sentiment_agent\"" in text

    def test_agent_failures_total(self) -> None:
        reg = MetricsRegistry(namespace="epic12_fail")
        reg.record_agent_failure("trend_follower", "timeout")
        reg.record_agent_failure("sentiment_agent", "api_error")
        text = reg.get_metrics_text().decode()
        assert 'agent_id="trend_follower",error_type="timeout"' in text
        assert 'agent_id="sentiment_agent",error_type="api_error"' in text
        assert "epic12_fail_agent_failures_total" in text

    def test_agent_schema_errors(self) -> None:
        reg = MetricsRegistry(namespace="epic12_schema")
        reg.record_agent_schema_error("trend_follower")
        reg.record_agent_schema_error("trend_follower")
        reg.record_agent_schema_error("sentiment_agent")
        text = reg.get_metrics_text().decode()
        assert "epic12_schema_agent_schema_errors_total{agent_id=\"trend_follower\"} 2" in text
        assert "epic12_schema_agent_schema_errors_total{agent_id=\"sentiment_agent\"} 1" in text


class TestEpic12AnalysisMetrics:
    """Test EPIC-12 analysis run/duration metrics."""

    def test_analysis_runs_total(self) -> None:
        reg = MetricsRegistry(namespace="epic12_analysis")
        reg.record_analysis_run("technical", duration=0.3)
        reg.record_analysis_run("fundamental", duration=2.5)
        reg.record_analysis_run("technical", duration=0.2)
        text = reg.get_metrics_text().decode()
        assert "epic12_analysis_analysis_runs_total{analysis_type=\"technical\"} 2" in text
        assert "epic12_analysis_analysis_runs_total{analysis_type=\"fundamental\"} 1" in text


class TestEpic12DataQualityMetrics:
    """Test EPIC-12 data quality and orderbook sequence gap metrics."""

    def test_data_quality_score(self) -> None:
        reg = MetricsRegistry(namespace="epic12_dq")
        reg.set_data_quality_score("binance_btc", 0.98)
        reg.set_data_quality_score("binance_eth", 0.95)
        text = reg.get_metrics_text().decode()
        assert 'epic12_dq_data_quality_score{data_source="binance_btc"} 0.98' in text

    def test_sequence_gaps(self) -> None:
        reg = MetricsRegistry(namespace="epic12_gaps")
        reg.record_sequence_gap("binance", "BTC/USDT")
        reg.record_sequence_gap("binance", "BTC/USDT")
        reg.record_sequence_gap("kraken", "ETH/USDT")
        text = reg.get_metrics_text().decode()
        # Prometheus sorts labels alphabetically: instrument before venue
        assert 'instrument="BTC/USDT",venue="binance"' in text
        assert 'instrument="ETH/USDT",venue="kraken"' in text
        # Verify counts: 2 binance, 1 kraken
        assert text.count('instrument="BTC/USDT",venue="binance"} 2.0') == 1
        assert text.count('instrument="ETH/USDT",venue="kraken"} 1.0') == 1


class TestEpic12ConsensusMetrics:
    """Test EPIC-12 consensus disagreement metrics."""

    def test_consensus_disagreement(self) -> None:
        reg = MetricsRegistry(namespace="epic12_consensus")
        reg.set_consensus_disagreement("5m", 0.3)
        reg.set_consensus_disagreement("5m", 0.1)  # update
        reg.set_consensus_disagreement("1h", 0.7)
        text = reg.get_metrics_text().decode()
        assert 'epic12_consensus_consensus_disagreement{window="5m"} 0.1' in text
        assert 'epic12_consensus_consensus_disagreement{window="1h"} 0.7' in text


class TestEpic12ForecastMetrics:
    """Test EPIC-12 forecast scoring metrics."""

    def test_forecast_brier(self) -> None:
        reg = MetricsRegistry(namespace="epic12_forecast")
        reg.set_forecast_brier("trend_follower", 0.15)
        reg.set_forecast_brier("sentiment_agent", 0.22)
        text = reg.get_metrics_text().decode()
        assert 'epic12_forecast_forecast_brier_score{agent_id="trend_follower"} 0.15' in text

    def test_forecast_log_loss(self) -> None:
        reg = MetricsRegistry(namespace="epic12_forecast")
        reg.set_forecast_log_loss("trend_follower", 0.35)
        text = reg.get_metrics_text().decode()
        assert 'epic12_forecast_forecast_log_loss{agent_id="trend_follower"} 0.35' in text


class TestEpic12PaperMetrics:
    """Test EPIC-12 paper trading metrics."""

    def test_paper_pnl(self) -> None:
        reg = MetricsRegistry(namespace="epic12_paper")
        reg.set_paper_pnl("realized", 1234.56)
        reg.set_paper_pnl("unrealized", -234.56)
        text = reg.get_metrics_text().decode()
        assert 'epic12_paper_paper_pnl{pnl_type="realized"} 1234.56' in text
        assert 'epic12_paper_paper_pnl{pnl_type="unrealized"} -234.56' in text

    def test_paper_drawdown(self) -> None:
        reg = MetricsRegistry(namespace="epic12_paper")
        reg.set_paper_drawdown(0.05)
        reg.set_paper_drawdown(0.12)
        text = reg.get_metrics_text().decode()
        assert 'epic12_paper_paper_drawdown 0.12' in text


class TestEpic12RiskMetrics:
    """Test EPIC-12 risk metrics."""

    def test_risk_blocks(self) -> None:
        reg = MetricsRegistry(namespace="epic12_risk")
        reg.record_risk_block("max_position_size")
        reg.record_risk_block("max_position_size")
        reg.record_risk_block("drawdown_limit")
        text = reg.get_metrics_text().decode()
        assert 'epic12_risk_risk_blocks_total{reason="max_position_size"} 2' in text
        assert 'epic12_risk_risk_blocks_total{reason="drawdown_limit"} 1' in text

    def test_no_trade_ratio(self) -> None:
        reg = MetricsRegistry(namespace="epic12_risk")
        reg.set_no_trade_ratio("trend_follower", 0.45)
        reg.set_no_trade_ratio("sentiment_agent", 0.30)
        text = reg.get_metrics_text().decode()
        assert 'epic12_risk_no_trade_ratio{agent_id="trend_follower"} 0.45' in text
        assert 'epic12_risk_no_trade_ratio{agent_id="sentiment_agent"} 0.3' in text


class TestEpic12Combined:
    """Test that all EPIC-12 metrics coexist in one registry."""

    def test_all_epic12_metrics_in_one_registry(self) -> None:
        reg = MetricsRegistry(namespace="epic12_all")

        # Agent metrics
        reg.record_agent_run("agent_a", "success", duration=0.5)
        reg.record_agent_failure("agent_a", "timeout")
        reg.record_agent_schema_error("agent_b")

        # Analysis metrics
        reg.record_analysis_run("technical", duration=0.3)

        # Data quality
        reg.set_data_quality_score("binance", 0.99)
        reg.record_sequence_gap("binance", "BTC/USDT")

        # Consensus
        reg.set_consensus_disagreement("5m", 0.2)

        # Forecast
        reg.set_forecast_brier("agent_a", 0.1)
        reg.set_forecast_log_loss("agent_a", 0.2)

        # Paper trading
        reg.set_paper_pnl("realized", 500.0)
        reg.set_paper_drawdown(0.03)

        # Risk
        reg.record_risk_block("max_position_size")
        reg.set_no_trade_ratio("agent_a", 0.4)

        text = reg.get_metrics_text().decode()

        # Verify all metric families present
        assert "epic12_all_agent_runs_total" in text
        assert "epic12_all_agent_failures_total" in text
        assert "epic12_all_agent_duration_seconds" in text
        assert "epic12_all_agent_schema_errors_total" in text
        assert "epic12_all_analysis_runs_total" in text
        assert "epic12_all_analysis_duration_seconds" in text
        assert "epic12_all_data_quality_score" in text
        assert "epic12_all_orderbook_sequence_gaps_total" in text
        assert "epic12_all_consensus_disagreement" in text
        assert "epic12_all_forecast_brier_score" in text
        assert "epic12_all_forecast_log_loss" in text
        assert "epic12_all_paper_pnl" in text
        assert "epic12_all_paper_drawdown" in text
        assert "epic12_all_risk_blocks_total" in text
        assert "epic12_all_no_trade_ratio" in text
