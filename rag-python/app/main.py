
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.api.routes.search import router as search_router


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="NexusAI — Document-based AI Knowledge Assistant",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


app.include_router(documents_router)
app.include_router(search_router)
@app.get("/", tags=["Health"])
async def root():
    return {
        "success": True,
        "message": "Welcome to NexusAI",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "success": True,
        "status": "healthy",
        "service": "nexus-ai-api",
    }
