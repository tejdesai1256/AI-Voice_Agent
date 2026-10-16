import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.llm_service import generate_response
from app.database.database import Base, engine
import app.database.models  # noqa: F401
from app.api.auth_routes import router as auth_router
from app.api.call_routes import router as call_router
from app.api.automation_routes import router as automation_router
from app.api.escalation_routes import router as escalation_router

app = FastAPI(title="AI Voice Calling Platform", version="2.0.0")

# Auto-initialize database tables if database is reachable
try:
    Base.metadata.create_all(bind=engine)
except Exception as db_err:
    print(f"[DB Warning] Table auto-creation deferred or failed: {db_err}")

# Ensure required directories exist
os.makedirs("recordings", exist_ok=True)
os.makedirs("static/waiting", exist_ok=True)

# Mount recordings audio directory
app.mount(
    "/audio",
    StaticFiles(directory="recordings", html=False),
    name="audio_files",
)

# Mount static directory for waiting audio and assets
app.mount(
    "/static",
    StaticFiles(directory="static", html=False),
    name="static_files",
)

# Include Routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(call_router, prefix="/api", tags=["Calls & Customers"])
app.include_router(automation_router, prefix="/api/automation", tags=["Automated Calling"])
app.include_router(escalation_router, prefix="/api/escalations", tags=["Human Assistance"])


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ai-test")
async def ai_test(prompt: str):
    response = await generate_response(prompt)
    return {
        "prompt": prompt,
        "response": response,
    }