import pytest
import pytest_asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import sys
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add the backend directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from session_manager import SessionManager


@pytest.fixture
def test_config():
    """Create a test configuration with safe defaults"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key-sk-ant-test123"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.CHROMA_PATH = "test_chroma_db"
    config.MAX_RESULTS = 3
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for unit testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Default mock behavior for search method
    mock_store.search.return_value = SearchResults(
        documents=["Sample course content about testing"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.5],
        error=None
    )
    
    mock_store.get_existing_course_titles.return_value = ["Test Course", "Advanced Topics"]
    mock_store.get_course_count.return_value = 2
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "lessons": [{"lesson_number": 1, "lesson_title": "Test Lesson"}]
        }
    ]
    
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful text response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response from Claude"
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response that includes tool usage"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_use = Mock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "search_course_content"
    mock_tool_use.id = "tool_call_123"
    mock_tool_use.input = {"query": "test query", "course_name": "Test Course"}
    
    mock_response.content = [mock_tool_use]
    
    return mock_response


@pytest.fixture
def sample_course():
    """Create a sample Course object for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction to Testing",
            lesson_link="https://example.com/lesson1"
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Testing Techniques", 
            lesson_link="https://example.com/lesson2"
        )
    ]
    
    return Course(
        title="Test Course",
        lessons=lessons,
        course_link="https://example.com/course"
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample CourseChunk objects for testing"""
    chunks = []
    for i, lesson in enumerate(sample_course.lessons):
        chunk = CourseChunk(
            course_title=sample_course.title,
            lesson_number=lesson.lesson_number,
            chunk_index=i,
            content=f"Test content for {lesson.title}"
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mocked vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create a CourseOutlineTool with mocked vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def ai_generator_mock(test_config, mock_anthropic_client):
    """Create an AIGenerator with mocked client"""
    generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
    generator.client = mock_anthropic_client
    return generator


@pytest.fixture
def mock_empty_search_results():
    """Create mock empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def mock_error_search_results():
    """Create mock search results with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Vector store connection failed"
    )


@pytest.fixture
def valid_anthropic_api_key():
    """Return a valid-looking test API key"""
    return "sk-ant-api03-test123-valid-format-key"


@pytest.fixture
def invalid_anthropic_api_key():
    """Return an invalid API key for testing"""
    return "invalid-key-123"


@pytest.fixture
def mock_rag_system(mock_vector_store, test_config):
    """Create a mock RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)
    mock_rag.session_manager = Mock(spec=SessionManager)
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager.clear_session.return_value = None
    
    # Mock query method
    mock_rag.query.return_value = (
        "This is a test response from the RAG system",
        ["Test Course - Lesson 1", "Test Course - Lesson 2"]
    )
    
    # Mock analytics method
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course", "Advanced Topics"]
    }
    
    return mock_rag


@pytest.fixture
def test_app():
    """Create a test FastAPI application without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Pydantic models for request/response
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
    
    app = FastAPI(title="Test RAG API")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mock RAG system will be injected in tests
    rag_system = None
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            
            answer, sources = rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    # Store reference for dependency injection in tests
    app.state.rag_system = None
    
    return app


@pytest.fixture
def test_client(mock_rag_system):
    """Create a test client with mocked dependencies"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Pydantic models for request/response
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
    
    app = FastAPI(title="Test RAG API")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Use the mock directly in route handlers
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
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

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    with TestClient(app) as client:
        yield client


# Helper functions for tests
def create_mock_search_results(documents: List[str], course_titles: List[str], lesson_numbers: List[int] = None) -> SearchResults:
    """Helper to create mock search results with given data"""
    if lesson_numbers is None:
        lesson_numbers = [1] * len(documents)
    
    metadata = []
    for i, (course_title, lesson_num) in enumerate(zip(course_titles, lesson_numbers)):
        metadata.append({
            "course_title": course_title,
            "lesson_number": lesson_num,
            "lesson_title": f"Lesson {lesson_num}"
        })
    
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=[0.5] * len(documents),
        error=None
    )