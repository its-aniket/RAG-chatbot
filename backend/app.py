from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import os

# Import authentication and security
try:
    from auth.firebase_auth import firebase_auth
    from auth.middleware import SecurityMiddleware, setup_rate_limiting
    security_available = True
except ImportError as e:
    print(f"Security modules not available: {e}")
    security_available = False

# Import database initialization
from database import init_database

# Import routes
from routes.documents import router as documents_router
from api.chat_routes import router as chat_router

# Import RAG routes (gracefully handle if dependencies not installed)
try:
    from routes.rag import router as rag_router
    rag_available = True
except ImportError:
    rag_router = None
    rag_available = False

app = FastAPI(title="RAG Chatbot Backend", version="1.0.0") # pyright: ignore[reportUnknownVariableType]

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        init_database()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        # Don't stop the app, just log the error

# Enable CORS for frontend communication
allowed_origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000", # Alternative localhost
]

# Add production origin if configured
if os.getenv("FRONTEND_URL"):
    allowed_origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add security middleware
if security_available:
    app.add_middleware(SecurityMiddleware, allowed_origins=allowed_origins)
    limiter = setup_rate_limiting(app)
    print("Security middleware enabled")
else:
    print("Security middleware disabled - install dependencies to enable")

# Include routers
app.include_router(documents_router)
app.include_router(chat_router)

# Include RAG router if available
if rag_available and rag_router:
    app.include_router(rag_router)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str

@app.get("/")
async def home():
    return {"message": "RAG Chatbot Backend API is running!"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="rag-chatbot-backend")

@app.get("/auth/status")
async def auth_status():
    """Get authentication system status"""
    if security_available:
        return {
            "authentication": "enabled",
            "firebase_status": firebase_auth.get_status()
        }
    else:
        return {
            "authentication": "disabled",
            "message": "Install firebase-admin to enable authentication"
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Placeholder response - will be replaced with actual RAG implementation
        response = ChatResponse(
            message=f"Echo: {request.message}",
            timestamp=datetime.now().isoformat()
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

