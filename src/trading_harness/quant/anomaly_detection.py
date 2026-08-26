"""Anomaly-Erkennung für OHLCV-Daten (Phase 3)."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class Anomaly:
    """Eine erkannte Anomalie."""
    timestamp: str
    symbol: str
    anomaly_type: str  # "price_shock", "volume_spike", "volatility_outlier"
    severity: float  # 0.0–1.0
    feature: str
    value: float
    zscore: float | None
    threshold: float


class AnomalyDetector:
    """Statistische Anomalieerkennung — nur stdlib."""

    def __init__(self, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5,
                 window_size: int = 20) -> None:
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.window_size = window_size

    def detect(self, candles: list[dict]) -> list[Anomaly]:
        """Detect anomalies across an entire candle series."""
        anomalies: list[Anomaly] = []
        if len(candles) < self.window_size + 1:
            return anomalies
        for i in range(self.window_size, len(candles)):
            history = candles[i - self.window_size:i]
            candle = candles[i]
            anomalies.extend(self.detect_single(candle, history))
        return anomalies

    def detect_single(self, candle: dict, history: list[dict]) -> list[Anomaly]:
        """Detect anomalies for one candle against its history window."""
        anomalies: list[Anomaly] = []
        ts = str(candle.get("time", ""))

        # Price shock: log return z-score
        if len(history) >= 2:
            returns = []
            for j in range(1, len(history)):
                prev_c = history[j-1].get("close", 0)
                curr_c = history[j].get("close", 0)
                if prev_c > 0 and curr_c > 0:
                    returns.append(math.log(curr_c / prev_c))
            if len(returns) >= 2:
                current_close = candle.get("close", 0)
                prev_close = history[-1].get("close", 0)
                if prev_close > 0 and current_close > 0:
                    current_return = math.log(current_close / prev_close)
                    mean_r = statistics.mean(returns)
                    std_r = statistics.pstdev(returns)
                    deviation = abs(current_return - mean_r)
                    if std_r > 0:
                        zscore = deviation / std_r
                    else:
                        # Flache Baseline (Varianz 0): jede Abweichung hat einen
                        # unbeschränkten Z-Score — als Max-Severity flaggen.
                        zscore = 0.0 if deviation == 0.0 else self.zscore_threshold * 2
                    if zscore > self.zscore_threshold:
                        severity = min(zscore / (self.zscore_threshold * 2), 1.0)
                        anomalies.append(Anomaly(
                            timestamp=ts, symbol="", anomaly_type="price_shock",
                            severity=severity, feature="close",
                            value=current_return, zscore=zscore,
                            threshold=self.zscore_threshold,
                        ))

        # Volume spike
        volumes = [h.get("volume", 0) for h in history if h.get("volume", 0) > 0]
        current_vol = candle.get("volume", 0)
        if len(volumes) >= 2 and current_vol > 0:
            mean_v = statistics.mean(volumes)
            std_v = statistics.pstdev(volumes)
            if mean_v > 0:
                vol_ratio = current_vol / mean_v
                relative_std = std_v / mean_v
                if relative_std > 0:
                    zscore_v = (vol_ratio - 1.0) / relative_std
                else:
                    # Wie oben: flache Volumen-Baseline, jede Abweichung = unbeschränkter Z-Score.
                    zscore_v = 0.0 if vol_ratio == 1.0 else self.zscore_threshold * 2
                if zscore_v > self.zscore_threshold:
                    severity = min(zscore_v / (self.zscore_threshold * 2), 1.0)
                    anomalies.append(Anomaly(
                        timestamp=ts, symbol="", anomaly_type="volume_spike",
                        severity=severity, feature="volume",
                        value=current_vol, zscore=zscore_v,
                        threshold=self.zscore_threshold,
                    ))

        # Volatility outlier (IQR)
        if len(history) >= self.window_size:
            log_returns = []
            for j in range(1, len(history)):
                pc = history[j-1].get("close", 0)
                cc = history[j].get("close", 0)
                if pc > 0 and cc > 0:
                    log_returns.append(math.log(cc / pc))
            if len(log_returns) >= 4:
                # Rolling volatility: std of log returns
                current_c = candle.get("close", 0)
                prev_c = history[-1].get("close", 0)
                if prev_c > 0 and current_c > 0:
                    extended = log_returns + [math.log(current_c / prev_c)]
                    current_vol_val = statistics.pstdev(extended[-self.window_size:])
                    # Historical volatilities
                    hist_vols = []
                    for k in range(self.window_size, len(log_returns) + 1):
                        window = log_returns[k - self.window_size:k]
                        if len(window) >= 2:
                            hist_vols.append(statistics.pstdev(window))
                    if len(hist_vols) >= 4:
                        hist_vols_sorted = sorted(hist_vols)
                        n = len(hist_vols_sorted)
                        q1 = hist_vols_sorted[n // 4]
                        q3 = hist_vols_sorted[3 * n // 4]
                        iqr = q3 - q1
                        upper_bound = q3 + self.iqr_multiplier * iqr
                        if current_vol_val > upper_bound > 0:
                            severity = min((current_vol_val - upper_bound) / upper_bound, 1.0)
                            anomalies.append(Anomaly(
                                timestamp=ts, symbol="", anomaly_type="volatility_outlier",
                                severity=severity, feature="volatility",
                                value=current_vol_val, zscore=None,
                                threshold=upper_bound,
                            ))
        return anomalies
