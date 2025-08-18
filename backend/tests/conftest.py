import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config

@pytest.fixture
def mock_config():
    """Mock configuration with proper settings"""
    config = Mock(spec=Config)
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 0  # This is the bug we're testing
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Mock successful search results
    mock_results = SearchResults(
        documents=["Sample course content about Python programming"],
        metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
        distances=[0.5]
    )
    mock_store.search.return_value = mock_results
    mock_store._resolve_course_name.return_value = "Python Basics"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    
    # Mock course metadata
    mock_store.get_all_courses_metadata.return_value = [{
        "title": "Python Basics",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction to Python"},
            {"lesson_number": 2, "lesson_title": "Variables and Data Types"}
        ]
    }]
    
    return mock_store

@pytest.fixture
def mock_empty_vector_store():
    """Mock vector store that returns empty results (simulating MAX_RESULTS=0 bug)"""
    mock_store = Mock(spec=VectorStore)
    
    # Mock empty search results (the bug)
    mock_results = SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )
    mock_store.search.return_value = mock_results
    mock_store._resolve_course_name.return_value = "Python Basics"
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generator"""
    mock_client = Mock()
    
    # Mock successful response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_anthropic_tool_use_client():
    """Mock Anthropic client that triggers tool use"""
    mock_client = Mock()
    
    # Mock response with tool use
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Create a mock tool_use block with properly configured attributes
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tool_1"
    mock_tool_block.name = "search_course_content"  # This will return the string directly
    mock_tool_block.input = {"query": "Python basics"}
    
    mock_response.content = [mock_tool_block]
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def sample_course_metadata():
    """Sample course metadata for testing"""
    return {
        "title": "Python Programming Course",
        "course_link": "https://example.com/python-course",
        "instructor": "John Doe",
        "lessons": [
            {
                "lesson_number": 1,
                "lesson_title": "Introduction to Python",
                "lesson_link": "https://example.com/lesson1"
            },
            {
                "lesson_number": 2,
                "lesson_title": "Variables and Data Types",
                "lesson_link": "https://example.com/lesson2"
            }
        ]
    }