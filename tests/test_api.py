from fastapi.testclient import TestClient

from trading_harness.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["live_execution_enabled"] is False
    assert data["kill_switch"] is True


def test_create_and_list_runs():
    client.post("/snapshots", json={
        "symbol": "BTCUSDT",
        "data": {"price": 50000},
    })
    resp = client.post("/runs", json={"snapshot_id": "snap-fake"})
    assert resp.status_code == 200
    run_data = resp.json()
    assert run_data["snapshot_id"] == "snap-fake"
    assert run_data["state"] == "CREATED"

    resp = client.get("/runs")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_get_run_not_found():
    resp = client.get("/runs/nonexistent")
    assert resp.status_code == 404


def test_transition_run():
    run = client.post("/runs", json={"snapshot_id": "snap-1"}).json()
    run_id = run["id"]

    resp = client.post(f"/runs/{run_id}/transition/DATA_READY")
    assert resp.status_code == 200
    assert resp.json()["state"] == "DATA_READY"

    resp = client.post(f"/runs/{run_id}/transition/ANALYSIS_RUNNING")
    assert resp.status_code == 200
    assert resp.json()["state"] == "ANALYSIS_RUNNING"


def test_invalid_transition():
    run = client.post("/runs", json={"snapshot_id": "snap-1"}).json()
    run_id = run["id"]

    resp = client.post(f"/runs/{run_id}/transition/COMPLETE")
    assert resp.status_code == 400


def test_add_decision():
    run = client.post("/runs", json={"snapshot_id": "snap-1"}).json()
    run_id = run["id"]

    resp = client.post(f"/runs/{run_id}/decision", json={
        "agent_id": "tech-1",
        "direction": "LONG",
        "confidence": 0.75,
    })
    assert resp.status_code == 200
    assert len(resp.json()["decisions"]) == 1


def test_complete_run():
    run = client.post("/runs", json={"snapshot_id": "snap-1"}).json()
    run_id = run["id"]

    resp = client.post(f"/runs/{run_id}/complete", json={
        "outcome": "NO_TRADE",
        "reason": "low_confensus",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "COMPLETE"
    assert data["outcome"] == "NO_TRADE"
    assert data["outcome_reason"] == "low_confensus"


def test_fail_run():
    run = client.post("/runs", json={"snapshot_id": "snap-1"}).json()
    run_id = run["id"]

    resp = client.post(f"/runs/{run_id}/fail", json={
        "error": "data invalid",
    })
    assert resp.status_code == 200
    assert resp.json()["state"] == "FAILED"
    assert resp.json()["error"] == "data invalid"


def test_record_performance():
    resp = client.post("/performance", json={
        "run_id": "run-1",
        "agent_id": "tech-1",
        "snapshot_id": "snap-1",
        "direction": "LONG",
        "confidence": 0.75,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run-1"
    assert data["direction"] == "LONG"


def test_performance_by_run():
    client.post("/performance", json={
        "run_id": "run-x", "agent_id": "a1", "snapshot_id": "s1",
        "direction": "LONG", "confidence": 0.5,
    })
    client.post("/performance", json={
        "run_id": "run-x", "agent_id": "a2", "snapshot_id": "s1",
        "direction": "SHORT", "confidence": 0.6,
    })
    resp = client.get("/performance/summary/run/run-x")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_performance_by_agent():
    client.post("/performance", json={
        "run_id": "r1", "agent_id": "tech-alpha", "snapshot_id": "s1",
        "direction": "LONG", "confidence": 0.5,
    })
    client.post("/performance", json={
        "run_id": "r2", "agent_id": "tech-alpha", "snapshot_id": "s2",
        "direction": "SHORT", "confidence": 0.4,
    })
    resp = client.get("/performance/summary/agent/tech-alpha")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_audit_log():
    client.post("/runs", json={"snapshot_id": "snap-a"})
    resp = client.get("/audit")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_kill_switch_toggle():
    assert client.get("/health").json()["kill_switch"] is True
    resp = client.post("/kill-switch/true")
    assert resp.json()["kill_switch"] is True
    resp = client.post("/kill-switch/false")
    assert resp.json()["kill_switch"] is False