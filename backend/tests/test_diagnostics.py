"""
Diagnostic tests for RAG system health checks.
These tests should be run first to identify basic system issues.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import anthropic
from anthropic import APIError, AuthenticationError

from config import Config, config
from vector_store import VectorStore
from ai_generator import AIGenerator


class TestAPIKeyValidation:
    """Test API key configuration and validation"""
    
    def test_api_key_exists_in_environment(self):
        """Check if ANTHROPIC_API_KEY exists in environment"""
        api_key = config.ANTHROPIC_API_KEY
        assert api_key is not None, "ANTHROPIC_API_KEY is not set"
        assert api_key != "", "ANTHROPIC_API_KEY is empty"
        assert len(api_key) > 10, "ANTHROPIC_API_KEY appears to be too short"
    
    def test_api_key_format_validation(self):
        """Validate API key has expected format"""
        api_key = config.ANTHROPIC_API_KEY
        
        # Basic format checks for Anthropic API keys
        if api_key and api_key != "":
            # Anthropic keys typically start with 'sk-ant-'
            if not api_key.startswith('sk-ant-'):
                pytest.skip("API key doesn't match expected Anthropic format - may be test key")
    
    @patch('anthropic.Anthropic')
    def test_api_connectivity_mock(self, mock_anthropic_class):
        """Test API connectivity with mock (safe test)"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Hello"
        mock_client.messages.create.return_value = mock_response
        
        # Test creating AI generator
        try:
            ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            ai_gen.client = mock_client  # Use mock client
            
            # Test basic generation
            response = ai_gen.generate_response("Hello")
            assert response == "Hello"
            
        except Exception as e:
            pytest.fail(f"Failed to create AIGenerator with mock: {e}")
    
    def test_api_key_authentication_format(self):
        """Test if API key can be used to create Anthropic client without error"""
        api_key = config.ANTHROPIC_API_KEY
        
        if not api_key or api_key == "":
            pytest.fail("No API key available for testing")
        
        try:
            # Just test client creation - don't make actual API calls
            client = anthropic.Anthropic(api_key=api_key)
            assert client is not None
        except Exception as e:
            pytest.fail(f"Failed to create Anthropic client: {e}")


class TestVectorStoreHealth:
    """Test vector store database health and accessibility"""
    
    def test_chroma_path_configuration(self):
        """Verify ChromaDB path is configured"""
        assert config.CHROMA_PATH is not None
        assert config.CHROMA_PATH != ""
        assert isinstance(config.CHROMA_PATH, str)
    
    def test_vector_store_creation(self):
        """Test vector store can be created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                store = VectorStore(temp_dir, config.EMBEDDING_MODEL, config.MAX_RESULTS)
                assert store is not None
                assert store.client is not None
                assert store.max_results == config.MAX_RESULTS
            except Exception as e:
                pytest.fail(f"Failed to create VectorStore: {e}")
            finally:
                # Explicitly cleanup ChromaDB resources to prevent file locks
                if 'store' in locals() and hasattr(store, 'client'):
                    try:
                        del store.client
                        del store
                    except:
                        pass
    
    def test_embedding_model_configuration(self):
        """Test embedding model is properly configured"""
        assert config.EMBEDDING_MODEL is not None
        assert config.EMBEDDING_MODEL != ""
        # Check if it's a valid model name format
        assert isinstance(config.EMBEDDING_MODEL, str)
    
    def test_vector_store_basic_operations(self):
        """Test basic vector store operations work"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = None
            try:
                store = VectorStore(temp_dir, config.EMBEDDING_MODEL, config.MAX_RESULTS)
                
                # Test getting existing course titles (should return empty list for new DB)
                titles = store.get_existing_course_titles()
                assert isinstance(titles, list)
                
                # Test getting course count (should return 0 for new DB)
                count = store.get_course_count()
                assert isinstance(count, int)
                assert count >= 0
                
            except Exception as e:
                pytest.fail(f"Vector store basic operations failed: {e}")
            finally:
                # Explicitly cleanup ChromaDB resources to prevent file locks
                if store is not None and hasattr(store, 'client'):
                    try:
                        del store.client
                        del store
                    except:
                        pass
    
    def test_existing_chroma_db_accessibility(self):
        """Test if existing ChromaDB database is accessible"""
        if not os.path.exists(config.CHROMA_PATH):
            pytest.skip("ChromaDB path doesn't exist - this is expected for fresh setup")
        
        try:
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            
            # Try to access existing data
            titles = store.get_existing_course_titles()
            count = store.get_course_count()
            
            print(f"Found {count} courses in existing database")
            print(f"Course titles: {titles}")
            
        except Exception as e:
            pytest.fail(f"Cannot access existing ChromaDB: {e}")


class TestDependenciesCheck:
    """Test that all required packages are properly installed"""
    
    def test_import_core_modules(self):
        """Test that all core modules can be imported"""
        try:
            import anthropic
            import chromadb
            import sentence_transformers
            import fastapi
            import uvicorn
            from dotenv import load_dotenv
        except ImportError as e:
            pytest.fail(f"Failed to import required module: {e}")
    
    def test_import_custom_modules(self):
        """Test that all custom modules can be imported"""
        try:
            from config import Config, config
            from vector_store import VectorStore
            from ai_generator import AIGenerator
            from search_tools import CourseSearchTool, ToolManager
            from rag_system import RAGSystem
            from models import Course, Lesson, CourseChunk
            from document_processor import DocumentProcessor
            from session_manager import SessionManager
        except ImportError as e:
            pytest.fail(f"Failed to import custom module: {e}")
    
    def test_configuration_loading(self):
        """Test that configuration loads without errors"""
        try:
            from config import config
            assert config is not None
            assert hasattr(config, 'ANTHROPIC_API_KEY')
            assert hasattr(config, 'ANTHROPIC_MODEL')
            assert hasattr(config, 'CHROMA_PATH')
        except Exception as e:
            pytest.fail(f"Configuration loading failed: {e}")
    
    def test_sentence_transformers_model(self):
        """Test that sentence transformers model can be loaded"""
        try:
            from sentence_transformers import SentenceTransformer
            # Don't actually load the model in tests to save time/resources
            # Just verify the class is available
            assert SentenceTransformer is not None
        except Exception as e:
            pytest.fail(f"SentenceTransformer import failed: {e}")


class TestDataIntegrityCheck:
    """Test data integrity and document loading"""
    
    def test_docs_folder_exists(self):
        """Verify documents folder exists"""
        docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
        docs_path = os.path.abspath(docs_path)
        
        if not os.path.exists(docs_path):
            pytest.skip(f"Docs folder doesn't exist at {docs_path} - this may be expected")
        
        assert os.path.isdir(docs_path), f"Docs path exists but is not a directory: {docs_path}"
    
    def test_docs_folder_has_files(self):
        """Check if docs folder contains course files"""
        docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
        docs_path = os.path.abspath(docs_path)
        
        if not os.path.exists(docs_path):
            pytest.skip("Docs folder doesn't exist")
        
        files = [f for f in os.listdir(docs_path) 
                if f.lower().endswith(('.pdf', '.docx', '.txt'))]
        
        if not files:
            pytest.skip("No course documents found in docs folder")
        
        print(f"Found {len(files)} course documents: {files}")
        assert len(files) > 0
    
    def test_document_processor_basic_functionality(self):
        """Test document processor can be created"""
        try:
            from document_processor import DocumentProcessor
            processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            assert processor is not None
        except Exception as e:
            pytest.fail(f"Failed to create DocumentProcessor: {e}")
    
    def test_existing_vector_data_integrity(self):
        """Test integrity of existing vector data if it exists"""
        if not os.path.exists(config.CHROMA_PATH):
            pytest.skip("No existing ChromaDB to check")
        
        try:
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            
            # Check if we have course metadata
            all_courses = store.get_all_courses_metadata()
            course_count = store.get_course_count()
            
            print(f"Vector store integrity check:")
            print(f"  Course count: {course_count}")
            print(f"  Metadata entries: {len(all_courses) if all_courses else 0}")
            
            if course_count > 0:
                # Test a basic search to ensure embeddings work
                results = store.search("test query")
                assert results is not None
                print(f"  Search test: {'PASS' if not results.error else 'FAIL - ' + results.error}")
                
        except Exception as e:
            pytest.fail(f"Vector data integrity check failed: {e}")


class TestSystemConfigurationHealth:
    """Test overall system configuration health"""
    
    def test_config_values_are_reasonable(self):
        """Test that configuration values are within reasonable ranges"""
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_SIZE < 10000, "CHUNK_SIZE seems too large"
        
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, "CHUNK_OVERLAP must be less than CHUNK_SIZE"
        
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be positive"
        assert config.MAX_RESULTS < 100, "MAX_RESULTS seems too large"
        
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY must be non-negative"
    
    def test_model_configuration(self):
        """Test AI model configuration"""
        assert config.ANTHROPIC_MODEL is not None
        assert config.ANTHROPIC_MODEL != ""
        assert "claude" in config.ANTHROPIC_MODEL.lower()
    
    def test_environment_file_loading(self):
        """Test that environment file can be loaded"""
        try:
            from dotenv import load_dotenv
            # This should work without errors
            load_dotenv()
        except Exception as e:
            pytest.fail(f"Failed to load environment file: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])