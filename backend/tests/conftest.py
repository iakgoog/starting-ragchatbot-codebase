import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

# Add the backend directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


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
        error=None,
    )

    mock_store.get_existing_course_titles.return_value = [
        "Test Course",
        "Advanced Topics",
    ]
    mock_store.get_course_count.return_value = 2
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "lessons": [{"lesson_number": 1, "lesson_title": "Test Lesson"}],
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
            lesson_link="https://example.com/lesson1",
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Testing Techniques",
            lesson_link="https://example.com/lesson2",
        ),
    ]

    return Course(
        title="Test Course", lessons=lessons, course_link="https://example.com/course"
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
            content=f"Test content for {lesson.title}",
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
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def mock_error_search_results():
    """Create mock search results with error"""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Vector store connection failed"
    )


@pytest.fixture
def valid_anthropic_api_key():
    """Return a valid-looking test API key"""
    return "sk-ant-api03-test123-valid-format-key"


@pytest.fixture
def invalid_anthropic_api_key():
    """Return an invalid API key for testing"""
    return "invalid-key-123"


# Helper functions for tests
def create_mock_search_results(
    documents: List[str], course_titles: List[str], lesson_numbers: List[int] = None
) -> SearchResults:
    """Helper to create mock search results with given data"""
    if lesson_numbers is None:
        lesson_numbers = [1] * len(documents)

    metadata = []
    for i, (course_title, lesson_num) in enumerate(zip(course_titles, lesson_numbers)):
        metadata.append(
            {
                "course_title": course_title,
                "lesson_number": lesson_num,
                "lesson_title": f"Lesson {lesson_num}",
            }
        )

    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=[0.5] * len(documents),
        error=None,
    )
