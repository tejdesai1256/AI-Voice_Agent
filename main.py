import os
from dotenv import load_dotenv

load_dotenv(override=True)
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.llm_service import generate_response
from app.database.database import Base, engine, ensure_schema_migrations
from app.database.models import SystemUser
import app.database.models  # noqa: F401
from app.api.auth_routes import router as auth_router
from app.api.call_routes import router as call_router
from app.api.automation_routes import router as automation_router
from app.api.escalation_routes import router as escalation_router
from app.api.telephony_routes import router as telephony_router
from app.utils.auth import get_current_user

app = FastAPI(title="AI Voice Calling Platform", version="2.0.0")

# Enable CORS for local dashboards and development servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-initialize database tables if database is reachable
try:
    Base.metadata.create_all(bind=engine)
    ensure_schema_migrations(engine)
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
app.include_router(telephony_router, prefix="/api/telephony", tags=["Telephony"])


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ai-test")
async def ai_test(prompt: str, current_user: SystemUser = Depends(get_current_user)):
    response = await generate_response(prompt)
    return {
        "prompt": prompt,
        "response": response,
    }