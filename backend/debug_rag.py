#!/usr/bin/env python3
"""
Debug script to test RAG system functionality.
This script tests the actual production RAG system to identify issues.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from config import config
from rag_system import RAGSystem


def test_basic_setup():
    """Test basic RAG system setup and configuration."""
    print("=== RAG System Debug ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    print("=== Configuration ===")
    print(f"API Key configured: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
    print(f"API Key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}")
    print(f"Model: {config.ANTHROPIC_MODEL}")
    print(f"ChromaDB path: {config.CHROMA_PATH}")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Max results: {config.MAX_RESULTS}")
    print()
    
    print("=== ChromaDB Status ===")
    chroma_path = Path(config.CHROMA_PATH)
    print(f"ChromaDB directory exists: {chroma_path.exists()}")
    if chroma_path.exists():
        files = list(chroma_path.iterdir())
        print(f"Files in ChromaDB: {[f.name for f in files]}")
    print()
    
    return config.ANTHROPIC_API_KEY is not None and config.ANTHROPIC_API_KEY.strip() != ""


def test_rag_initialization():
    """Test RAG system initialization."""
    print("=== RAG System Initialization ===")
    try:
        rag = RAGSystem(config)
        print("[OK] RAG system initialized successfully")
        return rag
    except Exception as e:
        print(f"[ERROR] RAG system initialization failed: {e}")
        return None


def test_vector_store_status(rag_system):
    """Test vector store status and data."""
    print("=== Vector Store Status ===")
    try:
        # Check course count
        course_count = rag_system.vector_store.get_course_count()
        print(f"Course count: {course_count}")
        
        # Check course titles
        course_titles = rag_system.vector_store.get_existing_course_titles()
        print(f"Course titles: {course_titles}")
        
        if course_count == 0:
            print("[WARNING] No courses found in vector store - this might be the issue!")
            
        return course_count > 0
    except Exception as e:
        print(f"[ERROR] Vector store check failed: {e}")
        return False


def test_tool_definitions(rag_system):
    """Test tool definitions."""
    print("=== Tool Definitions ===")
    try:
        tools = rag_system.tool_manager.get_tool_definitions()
        print(f"Number of tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description'][:50]}...")
        return True
    except Exception as e:
        print(f"[ERROR] Tool definitions check failed: {e}")
        return False


def test_simple_query(rag_system):
    """Test a simple query without using real API."""
    print("=== Simple Query Test ===")
    try:
        # Test tool execution directly
        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )
        print(f"Tool execution result: {result[:100]}...")
        
        if "No relevant content found" in result:
            print("[WARNING] Tool executed but found no content - vector store might be empty")
        else:
            print("[OK] Tool execution successful with content")
            
        return True
    except Exception as e:
        print(f"[ERROR] Tool execution failed: {e}")
        return False


def test_documents_loading():
    """Test if documents can be loaded."""
    print("=== Documents Loading Test ===")
    docs_path = Path("../docs")
    print(f"Docs directory: {docs_path.absolute()}")
    print(f"Docs directory exists: {docs_path.exists()}")
    
    if docs_path.exists():
        doc_files = [f for f in docs_path.iterdir() 
                    if f.suffix.lower() in ['.pdf', '.docx', '.txt']]
        print(f"Document files found: {[f.name for f in doc_files]}")
        return len(doc_files) > 0
    return False


def main():
    """Run all debug tests."""
    print("Starting RAG system debug...\n")
    
    # Test 1: Basic setup
    has_api_key = test_basic_setup()
    
    # Test 2: RAG initialization
    rag_system = test_rag_initialization()
    if not rag_system:
        print("[ERROR] Cannot proceed - RAG system failed to initialize")
        return
    
    # Test 3: Vector store status
    has_data = test_vector_store_status(rag_system)
    
    # Test 4: Tool definitions
    tools_ok = test_tool_definitions(rag_system)
    
    # Test 5: Simple query
    query_ok = test_simple_query(rag_system)
    
    # Test 6: Documents
    docs_available = test_documents_loading()
    
    print("\n=== Summary ===")
    print(f"API Key: {'[OK]' if has_api_key else '[ERROR]'}")
    print(f"Vector Store Data: {'[OK]' if has_data else '[ERROR]'}")
    print(f"Tool Definitions: {'[OK]' if tools_ok else '[ERROR]'}")
    print(f"Query Execution: {'[OK]' if query_ok else '[ERROR]'}")
    print(f"Documents Available: {'[OK]' if docs_available else '[ERROR]'}")
    
    print("\n=== Likely Issues ===")
    if not has_api_key:
        print("[ERROR] Missing or invalid API key - create .env file with ANTHROPIC_API_KEY")
    if not has_data:
        print("[ERROR] No course data in vector store - run document loading process")
    if not docs_available:
        print("[ERROR] No document files found - add course documents to docs/ folder")
    
    if has_api_key and has_data and tools_ok and query_ok:
        print("[OK] System appears to be working correctly!")
        print("   The 'query failed' issue might be in the API layer or frontend.")


if __name__ == "__main__":
    main()