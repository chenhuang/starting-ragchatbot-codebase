import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool
from vector_store import SearchResults

class TestCourseSearchTool:
    """Test cases for CourseSearchTool functionality"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
    
    def test_execute_with_valid_query_success(self, mock_vector_store):
        """Test execute() with valid query returns formatted results"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.3]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = tool.execute("What is Python?")
        
        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert "[Python Basics - Lesson 1]" in result
        assert "Python is a programming language" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Python Basics - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
    
    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execute() with course name filter"""
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Course content"],
            metadata=[{"course_title": "Python Advanced", "lesson_number": 2}],
            distances=[0.2]
        )
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute("functions", course_name="Python Advanced")
        
        mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name="Python Advanced",
            lesson_number=None
        )
        
        assert "Course content" in result
    
    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execute() with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Lesson content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute("loops", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="loops",
            course_name=None,
            lesson_number=3
        )
        
        assert "Lesson content" in result
    
    def test_execute_with_no_results(self, mock_empty_vector_store):
        """Test execute() when search returns no results (simulating MAX_RESULTS=0 bug)"""
        tool = CourseSearchTool(mock_empty_vector_store)
        
        # Mock empty results (this is the bug we're testing)
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_empty_vector_store.search.return_value = empty_results
        
        result = tool.execute("nonexistent topic")
        
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0
    
    def test_execute_with_no_results_and_filters(self, mock_empty_vector_store):
        """Test execute() with no results and filter information"""
        tool = CourseSearchTool(mock_empty_vector_store)
        
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_empty_vector_store.search.return_value = empty_results
        
        result = tool.execute("topic", course_name="Missing Course", lesson_number=5)
        
        assert "No relevant content found in course 'Missing Course' in lesson 5" in result
    
    def test_execute_with_search_error(self, mock_vector_store):
        """Test execute() when vector store returns an error"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock error result
        error_results = SearchResults(
            documents=[], 
            metadata=[], 
            distances=[], 
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = error_results
        
        result = tool.execute("any query")
        
        assert result == "Database connection failed"
    
    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test _format_results() when lesson number is not available"""
        tool = CourseSearchTool(mock_vector_store)
        
        results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "General Course"}],  # No lesson_number
            distances=[0.4]
        )
        
        formatted = tool._format_results(results)
        
        assert "[General Course]" in formatted
        assert "Content without lesson" in formatted
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "General Course"
        assert tool.last_sources[0]["link"] is None
    
    def test_format_results_multiple_documents(self, mock_vector_store):
        """Test _format_results() with multiple search results"""
        tool = CourseSearchTool(mock_vector_store)
        
        results = SearchResults(
            documents=["First result", "Second result"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/course-a-lesson1",
            "https://example.com/course-b-lesson2"
        ]
        
        formatted = tool._format_results(results)
        
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted
        assert "First result" in formatted
        assert "Second result" in formatted
        assert len(tool.last_sources) == 2
    
    def test_sources_tracking_and_reset(self, mock_vector_store):
        """Test that last_sources is properly tracked and can be reset"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Initially empty
        assert tool.last_sources == []
        
        # Execute search
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://test.com"
        
        tool.execute("test query")
        
        # Sources should be tracked
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        
        # Reset sources
        tool.last_sources = []
        assert tool.last_sources == []