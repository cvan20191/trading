import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import catalysts as catalysts_router
from app.routers import dashboard_state as dashboard_state_router
from app.routers import live_playbook as live_playbook_router
from app.routers import playbook as playbook_router
from app.routers import replay as replay_router
from app.routers import summary as summary_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Macro Playbook Dashboard API",
    description="LLM-powered macro regime summary engine.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summary_router.router)
app.include_router(dashboard_state_router.router)
app.include_router(playbook_router.router)
app.include_router(live_playbook_router.router)
app.include_router(catalysts_router.router)
app.include_router(replay_router.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
