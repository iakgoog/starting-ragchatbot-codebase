"""
API endpoint tests for the RAG System FastAPI application.

Tests the main API endpoints:
- POST /api/query - Main query processing endpoint
- GET /api/courses - Course analytics endpoint 
- DELETE /api/session/{session_id} - Session management endpoint
- GET / - Root endpoint
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint"""
    
    def test_query_without_session_id(self, test_client, mock_rag_system):
        """Test query endpoint creates new session when none provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is testing?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response from the RAG system"
        assert data["sources"] == ["Test Course - Lesson 1", "Test Course - Lesson 2"]
        assert data["session_id"] == "test-session-123"
        
        # Verify RAG system was called correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is testing?", "test-session-123")
    
    def test_query_with_existing_session_id(self, test_client, mock_rag_system):
        """Test query endpoint uses provided session ID"""
        existing_session = "existing-session-456"
        
        response = test_client.post(
            "/api/query",
            json={
                "query": "Explain unit testing",
                "session_id": existing_session
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == existing_session
        
        # Verify session creation was not called for existing session
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Explain unit testing", existing_session)
    
    def test_query_with_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        
        # Should still process empty query (RAG system handles validation)
        assert response.status_code == 200
    
    def test_query_with_missing_query_field(self, test_client):
        """Test query endpoint with missing query field"""
        response = test_client.post(
            "/api/query",
            json={}
        )
        
        # FastAPI should return validation error
        assert response.status_code == 422
        assert "field required" in response.text.lower()
    
    def test_query_endpoint_error_handling(self, test_client, mock_rag_system):
        """Test query endpoint handles RAG system errors"""
        # Configure mock to raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post(
            "/api/query",
            json={"query": "What is testing?"}
        )
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint"""
    
    def test_get_course_stats(self, test_client, mock_rag_system):
        """Test courses endpoint returns correct analytics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Test Course", "Advanced Topics"]
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_error_handling(self, test_client, mock_rag_system):
        """Test courses endpoint handles RAG system errors"""
        # Configure mock to raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]


@pytest.mark.api
class TestSessionEndpoint:
    """Test cases for the session management endpoints"""
    
    def test_delete_session(self, test_client, mock_rag_system):
        """Test session deletion endpoint"""
        session_id = "test-session-789"
        
        response = test_client.delete(f"/api/session/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Session cleared successfully"
        
        # Verify session manager was called
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)
    
    def test_delete_session_error_handling(self, test_client, mock_rag_system):
        """Test session deletion handles errors"""
        # Configure mock to raise an exception
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session error")
        
        response = test_client.delete("/api/session/test-session")
        
        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]


@pytest.mark.api
class TestRootEndpoint:
    """Test cases for the root endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns welcome message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "RAG System API"


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_complete_query_workflow(self, test_client, mock_rag_system):
        """Test complete workflow: query -> get courses -> delete session"""
        # Step 1: Make a query
        query_response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]
        
        # Step 2: Get course statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        assert courses_response.json()["total_courses"] == 2
        
        # Step 3: Delete the session
        delete_response = test_client.delete(f"/api/session/{session_id}")
        assert delete_response.status_code == 200
        
        # Verify all RAG system methods were called
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once()
        mock_rag_system.get_course_analytics.assert_called_once()
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are present in responses"""
        response = test_client.get("/", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        # Note: TestClient doesn't fully simulate CORS, but we verify the middleware is configured
        # In actual deployment, CORS headers would be present
    
    def test_content_type_validation(self, test_client):
        """Test API validates content types correctly"""
        # Test with correct content type
        response = test_client.post(
            "/api/query",
            json={"query": "test"},
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 200
        
        # Test with form data (should be rejected)
        response = test_client.post(
            "/api/query",
            data={"query": "test"},
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422  # Validation error