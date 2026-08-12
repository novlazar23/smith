"""Property-Tests für AnalysisMode — LIVE ist strukturell blockiert."""

import pytest
from packages.schemas.analysis_request import AnalysisMode


def test_live_mode_raises_value_error():
    """AnalysisMode("live") wirft einen ValueError."""
    with pytest.raises(ValueError):
        AnalysisMode("live")


def test_live_not_in_enum():
    """'live' ist nicht in den Werten von AnalysisMode enthalten."""
    assert "live" not in [m.value for m in AnalysisMode]
