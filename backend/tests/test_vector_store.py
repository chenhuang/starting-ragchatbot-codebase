import pytest
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults

class TestVectorStore:
    """Test cases for VectorStore functionality, focusing on MAX_RESULTS bug"""
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init_with_max_results_zero_bug(self, mock_embedding_fn, mock_chroma_client):
        """Test VectorStore initialization with MAX_RESULTS=0 (the bug)"""
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        # This simulates the bug: MAX_RESULTS=0
        vector_store = VectorStore("./test_db", "test-model", max_results=0)
        
        assert vector_store.max_results == 0  # This is the problematic configuration
        mock_chroma_client.assert_called_once()
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_max_results_zero_returns_empty(self, mock_embedding_fn, mock_chroma_client):
        """Test that MAX_RESULTS=0 causes search to return no results"""
        # Setup mock ChromaDB
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        # Mock ChromaDB query to return empty results when n_results=0
        mock_collection.query.return_value = {
            'documents': [[]],  # Empty results due to n_results=0
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Create vector store with MAX_RESULTS=0 (the bug)
        vector_store = VectorStore("./test_db", "test-model", max_results=0)
        
        # Perform search
        results = vector_store.search("Python functions")
        
        # Verify ChromaDB was called with n_results=0
        mock_collection.query.assert_called_once_with(
            query_texts=["Python functions"],
            n_results=0,  # This is the bug!
            where=None
        )
        
        # Verify results are empty due to the bug
        assert results.is_empty()
        assert len(results.documents) == 0
        assert len(results.metadata) == 0
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_proper_max_results_returns_data(self, mock_embedding_fn, mock_chroma_client):
        """Test that proper MAX_RESULTS value allows search to return results"""
        # Setup mock ChromaDB
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        # Mock ChromaDB query to return actual results when n_results > 0
        mock_collection.query.return_value = {
            'documents': [["Python is a programming language", "Functions in Python"]],
            'metadatas': [[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ]],
            'distances': [[0.1, 0.2]]
        }
        
        # Create vector store with proper MAX_RESULTS (the fix)
        vector_store = VectorStore("./test_db", "test-model", max_results=5)
        
        # Perform search
        results = vector_store.search("Python functions")
        
        # Verify ChromaDB was called with proper n_results
        mock_collection.query.assert_called_once_with(
            query_texts=["Python functions"],
            n_results=5,  # This is the fix!
            where=None
        )
        
        # Verify results contain data
        assert not results.is_empty()
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert "Python is a programming language" in results.documents
        assert "Functions in Python" in results.documents
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_explicit_limit_overrides_max_results(self, mock_embedding_fn, mock_chroma_client):
        """Test that explicit limit parameter overrides MAX_RESULTS"""
        # Setup mock ChromaDB
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        mock_collection.query.return_value = {
            'documents': [["Result 1", "Result 2", "Result 3"]],
            'metadatas': [[{"course_title": "Test"}, {"course_title": "Test"}, {"course_title": "Test"}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        
        # Create vector store with MAX_RESULTS=0 (bug) but override with limit
        vector_store = VectorStore("./test_db", "test-model", max_results=0)
        
        # Search with explicit limit
        results = vector_store.search("test query", limit=3)
        
        # Verify explicit limit was used instead of max_results
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,  # Explicit limit overrides MAX_RESULTS=0
            where=None
        )
        
        assert len(results.documents) == 3
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_name_filter(self, mock_embedding_fn, mock_chroma_client):
        """Test search with course name filtering"""
        # Setup mocks
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client = Mock()
        
        def mock_get_or_create(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            elif name == "course_content":
                return mock_content
            return Mock()
        
        mock_client.get_or_create_collection.side_effect = mock_get_or_create
        mock_chroma_client.return_value = mock_client
        
        # Mock course name resolution
        mock_catalog.query.return_value = {
            'documents': [["Python Programming"]],
            'metadatas': [[{"title": "Python Programming Course"}]]
        }
        
        # Mock content search results
        mock_content.query.return_value = {
            'documents': [["Python content"]],
            'metadatas': [[{"course_title": "Python Programming Course", "lesson_number": 1}]],
            'distances': [[0.1]]
        }
        
        vector_store = VectorStore("./test_db", "test-model", max_results=5)
        results = vector_store.search("functions", course_name="Python")
        
        # Verify course resolution was attempted
        mock_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
        
        # Verify content search used resolved course title in filter
        mock_content.query.assert_called_once_with(
            query_texts=["functions"],
            n_results=5,
            where={"course_title": "Python Programming Course"}
        )
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_lesson_number_filter(self, mock_embedding_fn, mock_chroma_client):
        """Test search with lesson number filtering"""
        # Setup mock ChromaDB
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        mock_collection.query.return_value = {
            'documents': [["Lesson 3 content"]],
            'metadatas': [[{"course_title": "Test Course", "lesson_number": 3}]],
            'distances': [[0.1]]
        }
        
        vector_store = VectorStore("./test_db", "test-model", max_results=5)
        results = vector_store.search("content", lesson_number=3)
        
        # Verify lesson filter was applied
        mock_collection.query.assert_called_once_with(
            query_texts=["content"],
            n_results=5,
            where={"lesson_number": 3}
        )
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_combined_filters(self, mock_embedding_fn, mock_chroma_client):
        """Test search with both course name and lesson number filters"""
        # Setup mocks similar to course name test
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client = Mock()
        
        def mock_get_or_create(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            elif name == "course_content":
                return mock_content
            return Mock()
        
        mock_client.get_or_create_collection.side_effect = mock_get_or_create
        mock_chroma_client.return_value = mock_client
        
        # Mock course resolution
        mock_catalog.query.return_value = {
            'documents': [["Advanced Python"]],
            'metadatas': [[{"title": "Advanced Python Course"}]]
        }
        
        # Mock content search
        mock_content.query.return_value = {
            'documents': [["Advanced content"]],
            'metadatas': [[{"course_title": "Advanced Python Course", "lesson_number": 5}]],
            'distances': [[0.1]]
        }
        
        vector_store = VectorStore("./test_db", "test-model", max_results=5)
        results = vector_store.search("advanced topics", course_name="Advanced", lesson_number=5)
        
        # Verify combined filter was applied
        expected_filter = {
            "$and": [
                {"course_title": "Advanced Python Course"},
                {"lesson_number": 5}
            ]
        }
        
        mock_content.query.assert_called_once_with(
            query_texts=["advanced topics"],
            n_results=5,
            where=expected_filter
        )
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_error_handling(self, mock_embedding_fn, mock_chroma_client):
        """Test search error handling"""
        # Setup mock to raise exception
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Database connection failed")
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        vector_store = VectorStore("./test_db", "test-model", max_results=5)
        results = vector_store.search("test query")
        
        # Verify error is captured in results
        assert results.error == "Search error: Database connection failed"
        assert results.is_empty()
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_course_name_resolution_failure(self, mock_embedding_fn, mock_chroma_client):
        """Test behavior when course name cannot be resolved"""
        # Setup mocks
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client = Mock()
        
        def mock_get_or_create(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            elif name == "course_content":
                return mock_content
            return Mock()
        
        mock_client.get_or_create_collection.side_effect = mock_get_or_create
        mock_chroma_client.return_value = mock_client
        
        # Mock empty course resolution (course not found)
        mock_catalog.query.return_value = {
            'documents': [[]],  # No courses found
            'metadatas': [[]]
        }
        
        vector_store = VectorStore("./test_db", "test-model", max_results=5)
        results = vector_store.search("content", course_name="NonexistentCourse")
        
        # Verify error message for unresolved course
        assert results.error == "No course found matching 'NonexistentCourse'"
        assert results.is_empty()
    
    def test_search_results_from_chroma_helper(self):
        """Test SearchResults.from_chroma() helper method"""
        chroma_results = {
            'documents': [["doc1", "doc2"]],
            'metadatas': [[{"meta1": "value1"}, {"meta2": "value2"}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"meta1": "value1"}, {"meta2": "value2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
        assert not results.is_empty()
    
    def test_search_results_empty_helper(self):
        """Test SearchResults.empty() helper method"""
        results = SearchResults.empty("No results found")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "No results found"
        assert results.is_empty()