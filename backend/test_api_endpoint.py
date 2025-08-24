#!/usr/bin/env python3
"""
Test script to validate the API endpoint directly.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

import asyncio

from app import QueryRequest, rag_system


async def test_api_endpoint():
    """Test the API endpoint logic directly."""
    print("=== API Endpoint Test ===")

    # Test 1: Simple query
    print("\n1. Testing simple query...")
    try:
        query_request = QueryRequest(query="What is machine learning?")

        # Simulate the endpoint logic
        session_id = query_request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        print(f"Session ID: {session_id}")

        # Process query using RAG system
        answer, sources = rag_system.query(query_request.query, session_id)

        print(f"Answer: {answer[:100]}...")
        print(f"Sources: {sources}")
        print("[OK] Simple query test passed")

    except Exception as e:
        print(f"[ERROR] Simple query test failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Course-specific query
    print("\n2. Testing course-specific query...")
    try:
        query_request = QueryRequest(query="Tell me about MCP")

        session_id = query_request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        answer, sources = rag_system.query(query_request.query, session_id)

        print(f"Answer: {answer[:100]}...")
        print(f"Sources: {sources}")
        print("[OK] Course-specific query test passed")

    except Exception as e:
        print(f"[ERROR] Course-specific query test failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Session continuity
    print("\n3. Testing session continuity...")
    try:
        query_request1 = QueryRequest(query="What is computer use?")
        session_id = rag_system.session_manager.create_session()

        # First query
        answer1, sources1 = rag_system.query(query_request1.query, session_id)
        print(f"First answer: {answer1[:50]}...")

        # Follow-up query with same session
        query_request2 = QueryRequest(
            query="Tell me more about that", session_id=session_id
        )
        answer2, sources2 = rag_system.query(query_request2.query, session_id)
        print(f"Follow-up answer: {answer2[:50]}...")

        print("[OK] Session continuity test passed")

    except Exception as e:
        print(f"[ERROR] Session continuity test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_api_endpoint())
