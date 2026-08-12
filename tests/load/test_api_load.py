"""
Load Testing — Trading Orchestra API Endpoints

Tests:
- GET /health
- GET /status
- GET /metrics
- POST /analyze
- POST /trade/signal

Ziele:
- 100 req/s
- p95 < 500ms
- Error Rate < 1%
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class EndpointStats:
    """Statistiken für einen einzelnen Endpunkt."""
    endpoint: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests * 100

    @property
    def p50(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        return sorted_times[len(sorted_times) // 2]

    @property
    def p95(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def avg(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def summary(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "error_rate_pct": round(self.error_rate, 2),
            "avg_ms": round(self.avg * 1000, 2),
            "p50_ms": round(self.p50 * 1000, 2),
            "p95_ms": round(self.p95 * 1000, 2),
            "p99_ms": round(self.p99 * 1000, 2),
        }


class LoadTestRunner:
    """Parallel Load Test Runner."""

    def __init__(self, base_url: str = "http://localhost:8080", max_connections: int = 50) -> None:
        self.base_url = base_url
        self.max_connections = max_connections
        self.stats: dict[str, EndpointStats] = {}

    async def run_test(
        self,
        endpoint: str,
        method: str = "GET",
        payload: dict | None = None,
        num_requests: int = 100,
        concurrency: int = 10,
    ) -> EndpointStats:
        """Führt einen Load Test für einen Endpunkt aus."""
        stats = EndpointStats(endpoint=endpoint, method=method)
        self.stats[endpoint] = stats

        semaphore = asyncio.Semaphore(concurrency)

        async def make_request() -> None:
            async with semaphore:
                stats.total_requests += 1
                async with httpx.AsyncClient(base_url=self.base_url, timeout=10) as client:
                    start = time.perf_counter()
                    try:
                        if method == "GET":
                            resp = await client.get(endpoint)
                        elif method == "POST":
                            resp = await client.post(endpoint, json=payload or {})
                        else:
                            raise ValueError(f"Unsupported method: {method}")

                        elapsed = time.perf_counter() - start
                        stats.response_times.append(elapsed)

                        if resp.status_code == 200:
                            stats.successful_requests += 1
                        else:
                            stats.failed_requests += 1
                    except Exception:
                        stats.failed_requests += 1

        # Run all requests
        tasks = [make_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)

        return stats

    def print_summary(self) -> None:
        """Gibt eine Zusammenfassung aller Load Tests aus."""
        print("\n" + "=" * 80)
        print("LOAD TEST SUMMARY")
        print("=" * 80)
        for _endpoint, stats in self.stats.items():
            summary = stats.summary()
            print(f"\n{stats.method} {stats.endpoint}")
            print(f"  Total Requests:    {summary['total_requests']}")
            print(f"  Successful:        {summary['successful']}")
            print(f"  Failed:            {summary['failed']}")
            print(f"  Error Rate:        {summary['error_rate_pct']}%")
            print(f"  Avg Response:      {summary['avg_ms']}ms")
            print(f"  p50 Response:      {summary['p50_ms']}ms")
            print(f"  p95 Response:      {summary['p95_ms']}ms")
            print(f"  p99 Response:      {summary['p99_ms']}ms")
        print("\n" + "=" * 80)


async def test_api_load() -> dict[str, dict]:
    """Haupt-Load Test für alle API Endpunkte."""
    runner = LoadTestRunner(base_url="http://localhost:8080", max_connections=50)

    # Run tests
    results = {}

    # Test 1: GET /health
    stats = await runner.run_test(
        endpoint="/health",
        method="GET",
        num_requests=50,
        concurrency=10,
    )
    results["/health"] = stats.summary()

    # Test 2: GET /status
    stats = await runner.run_test(
        endpoint="/status",
        method="GET",
        num_requests=50,
        concurrency=10,
    )
    results["/status"] = stats.summary()

    # Test 3: GET /metrics
    stats = await runner.run_test(
        endpoint="/metrics",
        method="GET",
        num_requests=50,
        concurrency=10,
    )
    results["/metrics"] = stats.summary()

    # Test 4: POST /analyze
    stats = await runner.run_test(
        endpoint="/analyze",
        method="POST",
        payload={
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "strategy": "macd_crossover",
        },
        num_requests=50,
        concurrency=10,
    )
    results["/analyze"] = stats.summary()

    # Test 5: POST /trade/signal
    stats = await runner.run_test(
        endpoint="/trade/signal",
        method="POST",
        payload={
            "symbol": "BTC/USDT",
            "action": "buy",
            "confidence": 0.75,
        },
        num_requests=50,
        concurrency=10,
    )
    results["/trade/signal"] = stats.summary()

    runner.print_summary()
    return results
