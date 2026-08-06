"""Feature extraction worker — computes ML features from OHLCV market data."""

from __future__ import annotations

from typing import Any

import numpy as np


class FeatureExtractor:
    """Extracts features from market data for ML models."""

    def extract(
        self,
        ohlcv: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extracts features from OHLCV data and returns a flat feature dict.

        Args:
            ohlcv: Dict with keys "open", "high", "low", "close", "volume",
                   each mapping to a 1-D numpy array.
            metadata: Optional metadata dict (unused in computation).

        Returns:
            Dict of {feature_name: float_value} using latest values.
        """
        close = np.asarray(ohlcv["close"], dtype=np.float64)
        open_ = np.asarray(ohlcv["open"], dtype=np.float64)
        high = np.asarray(ohlcv["high"], dtype=np.float64)
        low = np.asarray(ohlcv["low"], dtype=np.float64)
        volume = np.asarray(ohlcv["volume"], dtype=np.float64)

        features: dict[str, float] = {}

        # Log returns
        if len(close) >= 2:
            log_returns = np.log(close[1:] / close[:-1])
            features["return"] = float(log_returns[-1])
            # Pad short data: compute rolling features with available length
            features["volatility_20"] = float(np.std(log_returns[-20:], ddof=1)) if len(log_returns) >= 20 else 0.0
            features["momentum_10"] = float(log_returns[-10] if len(log_returns) >= 10 else log_returns[-1]) if len(log_returns) > 0 else 0.0
        else:
            features["return"] = 0.0
            features["volatility_20"] = 0.0
            features["momentum_10"] = 0.0

        # Volume ratio
        if len(volume) >= 21:
            avg_volume_20 = float(np.mean(volume[-21:-1]))
            features["volume_ratio"] = float(volume[-1] / avg_volume_20) if avg_volume_20 > 0 else 0.0
        elif len(volume) >= 2:
            features["volume_ratio"] = float(volume[-1] / volume[-2]) if volume[-2] > 0 else 0.0
        else:
            features["volume_ratio"] = 0.0

        # Price change and range
        if len(close) > 0:
            features["price_change"] = float(close[-1] - open_[-1])
            features["price_range"] = float(high[-1] - low[-1])
        else:
            features["price_change"] = 0.0
            features["price_range"] = 0.0

        # Volume change
        if len(volume) >= 2:
            features["volume_change"] = float(volume[-1] / volume[-2]) if volume[-2] > 0 else 0.0
        else:
            features["volume_change"] = 0.0

        return features

    def extract_rolling_features(
        self,
        ohlcv: dict[str, np.ndarray],
        window: int = 20,
    ) -> dict[str, float]:
        """Computes rolling statistics over a sliding window.

        Args:
            ohlcv: Dict with OHLCV numpy arrays.
            window: Rolling window size.

        Returns:
            Dict with mean_return, std_return, max_drawdown, volume_avg, volume_std.
        """
        close = np.asarray(ohlcv["close"], dtype=np.float64)
        volume = np.asarray(ohlcv["volume"], dtype=np.float64)

        features: dict[str, float] = {}

        if len(close) >= 2:
            log_returns = np.log(close[1:] / close[:-1])
            log_returns = np.where(np.isfinite(log_returns), log_returns, 0.0)
        else:
            log_returns = np.array([], dtype=np.float64)

        # Rolling returns stats
        if len(log_returns) >= window:
            rolling_rets = log_returns[-window:]
        elif len(log_returns) > 0:
            rolling_rets = log_returns
        else:
            rolling_rets = np.array([], dtype=np.float64)

        features["mean_return"] = float(np.mean(rolling_rets)) if len(rolling_rets) > 0 else 0.0
        features["std_return"] = float(np.std(rolling_rets, ddof=1)) if len(rolling_rets) > 1 else 0.0

        # Max drawdown from equity-like series
        if len(rolling_rets) >= 2:
            cumulative = np.cumprod(1.0 + rolling_rets)
            running_max = np.maximum.accumulate(cumulative)
            dd = (running_max - cumulative) / running_max
            dd = np.where(running_max == 0, 0.0, dd)
            features["max_drawdown"] = float(np.max(dd))
        else:
            features["max_drawdown"] = 0.0

        # Volume stats
        if len(volume) >= window:
            rolling_vol = volume[-window:]
        elif len(volume) > 0:
            rolling_vol = volume
        else:
            rolling_vol = np.array([], dtype=np.float64)

        features["volume_avg"] = float(np.mean(rolling_vol)) if len(rolling_vol) > 0 else 0.0
        features["volume_std"] = float(np.std(rolling_vol, ddof=1)) if len(rolling_vol) > 1 else 0.0

        return features

    def normalize_features(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Normalizes features to 0-1 range using min-max normalization.

        Uses the full set of feature values to determine min/max.

        Args:
            features: Dict of {feature_name: float_value}.

        Returns:
            Dict with normalized values in [0.0, 1.0].
        """
        if not features:
            return {}

        values = np.array(list(features.values()), dtype=np.float64)
        feat_min = float(np.min(values))
        feat_max = float(np.max(values))

        range_val = feat_max - feat_min
        if range_val == 0.0:
            return dict.fromkeys(features, 0.0)

        return {k: float((v - feat_min) / range_val) for k, v in features.items()}

    @staticmethod
    def _pad_to_max(arr: np.ndarray, max_len: int) -> np.ndarray:
        """Pads an array with zeros at the end up to max_len."""
        if len(arr) >= max_len:
            return arr[-max_len:]
        padded = np.zeros(max_len, dtype=arr.dtype)
        padded[: len(arr)] = arr
        return padded
