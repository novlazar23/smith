from fastapi import FastAPI

from trading_harness.api.routes import router

app = FastAPI(
    title="Evolutionary Trading Harness",
    version="0.1.0",
    description="Shadow-first evolutionary multi-agent trading research service",
)
app.include_router(router)
