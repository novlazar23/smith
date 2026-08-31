from pathlib import Path

from trading_harness.main import app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_web_proxy_strips_api_prefix() -> None:
    nginx_config = (REPOSITORY_ROOT / "web" / "nginx.conf").read_text()

    assert "location /api/" in nginx_config
    assert "proxy_pass http://api:8080/;" in nginx_config


def test_dashboard_handles_non_array_agent_responses() -> None:
    dashboard = (
        REPOSITORY_ROOT / "web" / "src" / "components" / "Dashboard.tsx"
    ).read_text()

    assert "const agents = Array.isArray(data) ? data : []" in dashboard


def test_quant_router_is_exposed_by_main_app() -> None:
    paths = set(app.openapi()["paths"])

    assert "/quant/status" in paths
    assert "/quant/backtest/run" in paths


def test_docker_image_installs_quant_dependencies() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text()

    assert "uv sync --frozen --no-dev --no-editable --extra quant" in dockerfile


def test_visible_dashboard_actions_call_the_api() -> None:
    components = REPOSITORY_ROOT / "web" / "src" / "components"
    agents = (components / "AgentList.tsx").read_text()
    shadow = (components / "ShadowTrading.tsx").read_text()
    backtest = (components / "BacktestView.tsx").read_text()

    assert "apiRequest<Agent>('/agents'" in agents
    assert "apiRequest(`/shadow-trading/${name}`" in shadow
    assert "apiRequest<BacktestResponse>('/quant/backtest/run'" in backtest
