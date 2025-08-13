"""
Integration tests for RAGSystem functionality.
Tests full query flow, tool manager integration, and error propagation.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component setup"""
    
    def test_rag_system_initialization(self, test_config):
        """Test RAGSystem initializes all components correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use temporary directory for testing
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Check all components are initialized
            assert rag_system.config == test_config
            assert rag_system.document_processor is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.session_manager is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None
            assert rag_system.outline_tool is not None
    
    def test_tool_registration(self, test_config):
        """Test that tools are properly registered with tool manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Check tools are registered
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            tool_names = [tool["name"] for tool in tool_definitions]
            
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
            assert len(tool_definitions) == 2
    
    def test_component_configuration_propagation(self, test_config):
        """Test that configuration values are properly passed to components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            test_config.MAX_RESULTS = 10
            test_config.MAX_HISTORY = 5
            
            rag_system = RAGSystem(test_config)
            
            # Check configuration propagation
            assert rag_system.vector_store.max_results == 10
            assert rag_system.session_manager.max_history == 5


class TestQueryProcessing:
    """Test end-to-end query processing"""
    
    @patch('anthropic.Anthropic')
    def test_simple_query_flow(self, mock_anthropic_class, test_config):
        """Test complete query flow without tools"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            # Setup mock AI response
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "This is a general knowledge answer"
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            
            rag_system = RAGSystem(test_config)
            
            # Execute query
            response, sources = rag_system.query("What is 2+2?")
            
            # Verify response
            assert response == "This is a general knowledge answer"
            assert sources == []  # No sources for general knowledge
            
            # Verify AI was called
            mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_tool_based_query_flow(self, mock_anthropic_class, test_config):
        """Test query flow that triggers tool usage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            # Setup mock AI client
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Setup tool use response
            tool_use_response = Mock()
            tool_use_response.stop_reason = "tool_use"
            
            tool_use_block = Mock()
            tool_use_block.type = "tool_use"
            tool_use_block.name = "search_course_content"
            tool_use_block.id = "tool_call_123"
            tool_use_block.input = {"query": "machine learning basics"}
            
            tool_use_response.content = [tool_use_block]
            
            # Setup final response
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = "Here's what I found about machine learning..."
            
            mock_client.messages.create.side_effect = [tool_use_response, final_response]
            
            rag_system = RAGSystem(test_config)
            
            # Mock the vector store search to return results
            mock_search_results = SearchResults(
                documents=["Machine learning is a subset of AI..."],
                metadata=[{"course_title": "AI Basics", "lesson_number": 1}],
                distances=[0.5],
                error=None
            )
            rag_system.vector_store.search = Mock(return_value=mock_search_results)
            rag_system.vector_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
            
            # Execute query
            response, sources = rag_system.query("Tell me about machine learning")
            
            # Verify response and sources
            assert response == "Here's what I found about machine learning..."
            assert len(sources) == 1
            assert sources[0]["text"] == "AI Basics - Lesson 1"
            assert sources[0]["link"] == "https://example.com/lesson1"
            
            # Verify two API calls were made (tool use + final response)
            assert mock_client.messages.create.call_count == 2
    
    def test_query_with_session_history(self, test_config):
        """Test query processing with conversation history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            with patch('anthropic.Anthropic') as mock_anthropic_class:
                mock_client = Mock()
                mock_anthropic_class.return_value = mock_client
                
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Contextual response"
                mock_response.stop_reason = "end_turn"
                mock_client.messages.create.return_value = mock_response
                
                rag_system = RAGSystem(test_config)
                
                # First query to establish history
                rag_system.query("Initial question", session_id="test_session")
                
                # Second query with history
                response, sources = rag_system.query("Follow-up question", session_id="test_session")
                
                # Verify history was used in second call
                second_call = mock_client.messages.create.call_args_list[1]
                system_content = second_call.kwargs["system"]
                assert "Previous conversation:" in system_content
    
    def test_query_without_session(self, test_config):
        """Test query processing without session ID"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            with patch('anthropic.Anthropic') as mock_anthropic_class:
                mock_client = Mock()
                mock_anthropic_class.return_value = mock_client
                
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Stateless response"
                mock_response.stop_reason = "end_turn"
                mock_client.messages.create.return_value = mock_response
                
                rag_system = RAGSystem(test_config)
                
                response, sources = rag_system.query("Test question")
                
                # Verify no history was used
                call_args = mock_client.messages.create.call_args
                system_content = call_args.kwargs["system"]
                assert "Previous conversation:" not in system_content


class TestDocumentManagement:
    """Test document adding and processing"""
    
    def test_add_course_document_success(self, test_config, sample_course):
        """Test successfully adding a course document"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Mock document processor
            mock_chunks = [
                CourseChunk(
                    course_title=sample_course.title,
                    lesson_number=1,
                    lesson_title="Test Lesson",
                    content="Test content",
                    metadata={"course_title": sample_course.title}
                )
            ]
            
            rag_system.document_processor.process_course_document = Mock(
                return_value=(sample_course, mock_chunks)
            )
            
            # Mock vector store methods
            rag_system.vector_store.add_course_metadata = Mock()
            rag_system.vector_store.add_course_content = Mock()
            
            # Add document
            course, chunk_count = rag_system.add_course_document("test_file.txt")
            
            # Verify results
            assert course == sample_course
            assert chunk_count == 1
            
            # Verify vector store was called
            rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
            rag_system.vector_store.add_course_content.assert_called_once_with(mock_chunks)
    
    def test_add_course_document_failure(self, test_config):
        """Test handling of document processing failure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Mock document processor to raise exception
            rag_system.document_processor.process_course_document = Mock(
                side_effect=Exception("File not found")
            )
            
            # Add document should handle error gracefully
            course, chunk_count = rag_system.add_course_document("nonexistent.txt")
            
            assert course is None
            assert chunk_count == 0
    
    def test_add_course_folder(self, test_config):
        """Test adding courses from a folder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            # Create test folder with documents
            docs_folder = os.path.join(temp_dir, "docs")
            os.makedirs(docs_folder)
            
            # Create test files
            test_files = ["course1.txt", "course2.pdf", "course3.docx", "readme.md"]
            for file_name in test_files:
                with open(os.path.join(docs_folder, file_name), "w") as f:
                    f.write("test content")
            
            rag_system = RAGSystem(test_config)
            
            # Mock vector store methods
            rag_system.vector_store.get_existing_course_titles = Mock(return_value=[])
            rag_system.vector_store.add_course_metadata = Mock()
            rag_system.vector_store.add_course_content = Mock()
            
            # Mock document processor
            mock_course = Course(title="Test Course", lessons=[], course_link="")
            mock_chunks = [CourseChunk("Test Course", 1, "Lesson", "Content", {})]
            
            rag_system.document_processor.process_course_document = Mock(
                return_value=(mock_course, mock_chunks)
            )
            
            # Add folder
            total_courses, total_chunks = rag_system.add_course_folder(docs_folder)
            
            # Should process 3 valid files (txt, pdf, docx) but skip .md
            assert total_courses == 3
            assert total_chunks == 3
    
    def test_skip_existing_courses(self, test_config):
        """Test that existing courses are skipped"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            docs_folder = os.path.join(temp_dir, "docs")
            os.makedirs(docs_folder)
            
            with open(os.path.join(docs_folder, "existing_course.txt"), "w") as f:
                f.write("content")
            
            rag_system = RAGSystem(test_config)
            
            # Mock that course already exists
            rag_system.vector_store.get_existing_course_titles = Mock(
                return_value=["Existing Course"]
            )
            
            mock_course = Course(title="Existing Course", lessons=[], course_link="")
            rag_system.document_processor.process_course_document = Mock(
                return_value=(mock_course, [])
            )
            
            # Add folder - should skip existing course
            total_courses, total_chunks = rag_system.add_course_folder(docs_folder)
            
            assert total_courses == 0
            assert total_chunks == 0


class TestErrorPropagation:
    """Test how errors bubble up through the system"""
    
    @patch('anthropic.Anthropic')
    def test_api_error_propagation(self, mock_anthropic_class, test_config):
        """Test that API errors are properly propagated"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")
            
            rag_system = RAGSystem(test_config)
            
            # Query should raise the API error
            with pytest.raises(Exception, match="API Error"):
                rag_system.query("Test question")
    
    def test_vector_store_error_propagation(self, test_config):
        """Test that vector store errors are handled in tool execution"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            with patch('anthropic.Anthropic') as mock_anthropic_class:
                # Setup tool use response
                mock_client = Mock()
                mock_anthropic_class.return_value = mock_client
                
                tool_use_response = Mock()
                tool_use_response.stop_reason = "tool_use"
                
                tool_use_block = Mock()
                tool_use_block.type = "tool_use"
                tool_use_block.name = "search_course_content"
                tool_use_block.id = "tool_call_123"
                tool_use_block.input = {"query": "test"}
                
                tool_use_response.content = [tool_use_block]
                
                final_response = Mock()
                final_response.content = [Mock()]
                final_response.content[0].text = "Error was handled"
                
                mock_client.messages.create.side_effect = [tool_use_response, final_response]
                
                rag_system = RAGSystem(test_config)
                
                # Mock vector store to return error
                error_results = SearchResults([], [], [], error="Vector store failed")
                rag_system.vector_store.search = Mock(return_value=error_results)
                
                # Query should complete despite vector store error
                response, sources = rag_system.query("Test question")
                
                assert response == "Error was handled"
                # The error should be passed to the AI in tool results
    
    def test_tool_manager_error_handling(self, test_config):
        """Test tool manager error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Test executing non-existent tool
            result = rag_system.tool_manager.execute_tool("nonexistent_tool", query="test")
            assert "Tool 'nonexistent_tool' not found" in result


class TestSourceTracking:
    """Test source tracking and management"""
    
    def test_source_tracking_and_reset(self, test_config):
        """Test that sources are tracked and reset correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Mock search results with sources
            mock_results = SearchResults(
                documents=["Test content"],
                metadata=[{"course_title": "Source Course", "lesson_number": 1}],
                distances=[0.5],
                error=None
            )
            
            rag_system.vector_store.search = Mock(return_value=mock_results)
            rag_system.vector_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
            
            # Execute search directly through tool
            rag_system.search_tool.execute("test query")
            
            # Check sources are tracked
            sources = rag_system.tool_manager.get_last_sources()
            assert len(sources) == 1
            assert sources[0]["text"] == "Source Course - Lesson 1"
            
            # Reset sources
            rag_system.tool_manager.reset_sources()
            
            # Check sources are cleared
            sources_after_reset = rag_system.tool_manager.get_last_sources()
            assert sources_after_reset == []


class TestAnalytics:
    """Test analytics and reporting functionality"""
    
    def test_get_course_analytics(self, test_config):
        """Test course analytics functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config.CHROMA_PATH = temp_dir
            
            rag_system = RAGSystem(test_config)
            
            # Mock vector store analytics methods
            rag_system.vector_store.get_course_count = Mock(return_value=5)
            rag_system.vector_store.get_existing_course_titles = Mock(
                return_value=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
            )
            
            analytics = rag_system.get_course_analytics()
            
            assert analytics["total_courses"] == 5
            assert len(analytics["course_titles"]) == 5
            assert "Course 1" in analytics["course_titles"]


class TestConfigurationIntegration:
    """Test configuration integration across components"""
    
    def test_configuration_consistency(self, test_config):
        """Test that configuration is consistently applied across all components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set specific test values
            test_config.CHROMA_PATH = temp_dir
            test_config.MAX_RESULTS = 7
            test_config.MAX_HISTORY = 3
            test_config.CHUNK_SIZE = 500
            test_config.CHUNK_OVERLAP = 50
            
            rag_system = RAGSystem(test_config)
            
            # Verify configuration propagation
            assert rag_system.vector_store.chroma_path == temp_dir
            assert rag_system.vector_store.max_results == 7
            assert rag_system.session_manager.max_history == 3
            assert rag_system.document_processor.chunk_size == 500
            assert rag_system.document_processor.chunk_overlap == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])