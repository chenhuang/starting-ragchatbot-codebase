import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json

class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""
    
    @pytest.mark.api
    def test_query_with_session_id(self, client, sample_query_request, mock_rag_system):
        """Test query endpoint with provided session ID"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)
        
        mock_rag_system.query.assert_called_once_with(
            sample_query_request["query"], 
            sample_query_request["session_id"]
        )
    
    @pytest.mark.api
    def test_query_without_session_id(self, client, sample_query_request_no_session, mock_rag_system):
        """Test query endpoint without session ID (should create new session)"""
        response = client.post("/api/query", json=sample_query_request_no_session)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once()
    
    @pytest.mark.api
    def test_query_invalid_request(self, client):
        """Test query endpoint with invalid request data"""
        invalid_request = {"invalid_field": "value"}
        
        response = client.post("/api/query", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_query_empty_query(self, client):
        """Test query endpoint with empty query string"""
        empty_query = {"query": ""}
        
        response = client.post("/api/query", json=empty_query)
        assert response.status_code == 200
        # Should still process empty query through RAG system

class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""
    
    @pytest.mark.api
    def test_get_courses_success(self, client, expected_course_stats, mock_rag_system):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    @pytest.mark.api
    def test_get_courses_with_rag_error(self, client, mock_rag_system):
        """Test courses endpoint when RAG system raises error"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")
        
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

class TestRootEndpoint:
    """Test cases for / (root) endpoint"""
    
    @pytest.mark.api
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Course Materials RAG System"

class TestRequestValidation:
    """Test request validation and error handling"""
    
    @pytest.mark.api
    def test_query_malformed_json(self, client):
        """Test query endpoint with malformed JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @pytest.mark.api 
    def test_query_missing_required_field(self, client):
        """Test query endpoint missing required query field"""
        incomplete_request = {"session_id": "test-123"}
        
        response = client.post("/api/query", json=incomplete_request)
        assert response.status_code == 422
        
        detail = response.json()["detail"]
        assert any("query" in str(error) for error in detail)

class TestContentTypes:
    """Test various content types and headers"""
    
    @pytest.mark.api
    def test_query_with_correct_content_type(self, client, sample_query_request):
        """Test query with proper JSON content type"""
        response = client.post(
            "/api/query",
            json=sample_query_request,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
    
    @pytest.mark.api
    def test_courses_response_headers(self, client):
        """Test that courses endpoint returns proper JSON response"""
        response = client.get("/api/courses")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

class TestResponseModels:
    """Test response model validation"""
    
    @pytest.mark.api
    def test_query_response_structure(self, client, sample_query_request):
        """Test that query response matches expected model structure"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
    
    @pytest.mark.api
    def test_courses_response_structure(self, client):
        """Test that courses response matches expected model structure"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Verify data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])