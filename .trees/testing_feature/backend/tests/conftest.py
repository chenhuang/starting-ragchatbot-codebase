import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os
import sys

# Add backend directory to Python path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing without dependencies"""
    mock_rag = Mock()
    mock_rag.query.return_value = ("Test answer", ["source1.pdf", "source2.pdf"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.add_course_folder.return_value = (2, 10)
    return mock_rag

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    mock_cfg = Mock()
    mock_cfg.anthropic_api_key = "test-key"
    mock_cfg.vector_store_path = "/tmp/test_vector_store"
    return mock_cfg

@pytest.fixture
def test_app(mock_rag_system, mock_config):
    """Create test FastAPI app with mocked dependencies"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app without static file mounting
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Test endpoints with mocked RAG system
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    return app

@pytest.fixture
def client(test_app):
    """Test client for API testing"""
    return TestClient(test_app)

@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123"
    }

@pytest.fixture
def sample_query_request_no_session():
    """Sample query request without session ID"""
    return {
        "query": "What is deep learning?"
    }

@pytest.fixture
def expected_query_response():
    """Expected query response structure"""
    return {
        "answer": "Test answer",
        "sources": ["source1.pdf", "source2.pdf"],
        "session_id": "test-session-123"
    }

@pytest.fixture
def expected_course_stats():
    """Expected course statistics response"""
    return {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }

@pytest.fixture
def temp_directory():
    """Temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress resource tracker warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")